"""
Simplified version of PhysVAE for ablation study.
TODO add wandb logging to check the balance between different terms.

NEW METRICS ADDED:
- residual_loss: L2 difference between raw physics output (x_P) and corrected output (x_PB)
- residual_rel_diff: Relative difference as percentage of raw output magnitude
  This helps monitor how much the correction layer is modifying the physics output.
  Lower values indicate the correction is making smaller changes.
  The residual_loss is computed as torch.sum((x_PB - x_P).pow(2), dim=1).mean().
"""
# Adapted from the training script of Phys-VAE
import numpy as np
import torch
import os
from torchvision.utils import make_grid
from base import BaseTrainer, PARENT_DIR
from utils import inf_loop, MetricTracker, kldiv_normal_normal
import wandb
from model.loss import mse_loss, mse_loss_per_channel
from IPython import embed

class PhysVAETrainerSMPL(BaseTrainer):
    """
    Trainer for Phys-VAE, with options for physics-based regularization and reconstruction.
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.device = device
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler

        self.no_phy = config['arch']['phys_vae']['no_phy']
        self.epochs_pretrain = config['trainer']['phys_vae'].get('epochs_pretrain', 0)
        self.dim_z_phy = config['arch']['phys_vae']['dim_z_phy']
        self.dim_z_aux = config['arch']['phys_vae']['dim_z_aux']

        # CHANGED: separate KL controls for z_phy and z_aux
        self.beta_warmup = config['trainer']['phys_vae'].get('kl_warmup_epochs', 50)  # NEW
        self.gate_loss_weight = config['trainer']['phys_vae'].get('balance_gate', 1e-3)  # NEW
        
        # Separate KL controls for z_phy
        self.use_kl_term_z_phy = config['trainer']['phys_vae'].get('use_kl_term_z_phy', False)
        self.beta_max_z_phy = config['trainer']['phys_vae'].get('beta_max_z_phy', 1.0)
        
        # Separate KL controls for z_aux
        self.use_kl_term_z_aux = config['trainer']['phys_vae'].get('use_kl_term_z_aux', False)
        self.beta_max_z_aux = config['trainer']['phys_vae'].get('beta_max_z_aux', 1.0)
        
        # ========================================================================
        # CAPACITY CONTROL MODULE (Optional add-on to original implementation)
        # ========================================================================
        self.use_capacity_control = config['trainer']['phys_vae'].get('use_capacity_control', False)
        self.C_max = config['trainer']['phys_vae'].get('C_max', float(self.dim_z_phy))  # e.g., 7.0 for RTM
        self.C_gamma = config['trainer']['phys_vae'].get('C_gamma', 10.0)  # Capacity penalty weight
        self.beta_aux = config['trainer']['phys_vae'].get('beta_aux', 1.0)  # Auxiliary KL weight

        # for Stage A bootstrap in u-space
        self.synthetic_data_loss_weight = config['trainer']['phys_vae'].get('balance_data_aug', 1.0)

        # === [REBUTTAL ONLY] not part of the published PILA runs ==============
        # Reviewer Q: "Could PILA be improved through addition of a physical
        # consistency loss as in HVAE?"  Keeps the HVAE-style synthetic-pair
        # inversion loss ON during the main phase, instead of only during the
        # Stage-A bootstrap. Configs under configs/ do not set this key, so it
        # defaults to 0.0 and the term is never computed -> published runs are
        # bit-identical. Used only by rebuttal/consistency_loss/configs/.
        # Note: this loss lives in u-space over dim_z_phy latents, so unlike
        # rec_loss its magnitude does not scale with the number of input bands.
        self.syn_cotrain_weight = config['trainer']['phys_vae'].get('balance_data_aug_cotrain', 0.0)
        # ======================================================================
        
        # NEW: Loss weights for improved residual architecture
        self.ortho_penalty_weight = config['trainer']['phys_vae'].get('ortho_penalty_weight', 0.1)
        self.coeff_penalty_weight = config['trainer']['phys_vae'].get('coeff_penalty_weight', 1e-4) #TODO: double check
        self.delta_penalty_weight = config['trainer']['phys_vae'].get('delta_penalty_weight', 1e-4) #TODO: double check
        
        # NEW: Edge penalty for z_phy to avoid extreme values (0, 1)
        self.edge_penalty_weight = config['trainer']['phys_vae'].get('edge_penalty_weight', 0.0)
        self.edge_penalty_power = config['trainer']['phys_vae'].get('edge_penalty_power', 1.0)
        
        # NEW: Temporal smoothness regularization for Mogi source parameters
        self.temporal_smoothness_weight = config['trainer']['phys_vae'].get('temporal_smoothness_weight', 0.0)
        
        # NEW: EMA prior configuration
        self.use_ema_prior = config['trainer']['phys_vae'].get('use_ema_prior', False)

        # NEW: gradient clipping
        self.grad_clip_norm = config['trainer'].get('grad_clip_norm', 1.0)

        # Reconstruction loss is summed over the feature axis, so its magnitude
        # scales with the number of channels (11 S2 bands vs 1721 hyperspectral
        # ones). Scaling it here keeps every other loss weight transferable
        # instead of re-tuning them per dataset. Default 1.0 = unchanged.
        self.rec_loss_scale = config['trainer'].get('rec_loss_scale', 1.0)

        # trackers
        self.train_metrics = MetricTracker(
            'loss', 'rec_loss', 'kl_loss',
            'syn_data_loss',  # diagnostic
            'residual_loss',  # L2 difference between raw and corrected output
            'residual_rel_diff',  # relative difference as percentage
            'ortho_penalty',  # orthogonality penalty for basis matrix
            'coeff_penalty',  # coefficient L2 penalty
            'delta_penalty',  # delta L2 penalty
            'edge_penalty',  # edge penalty for z_phy
            'temporal_smoothness',  # temporal smoothness penalty for Mogi source
            'c_norm',  # norm of coefficient vector
            'delta_norm',  # norm of residual vector
            's_norm',  # norm of scale parameters
            'basis_quality',  # ||B^T B - I||_F
            # NEW: Capacity control metrics
            'kl_u_phy',  # Physics KL in u-space
            'kl_z_aux',  # Auxiliary KL
            'beta_z_phy',  # Beta weight for z_phy KL term
            'beta_z_aux',  # Beta weight for z_aux KL term
            'capacity_target',  # Current capacity target C_t
            'capacity_penalty',  # Capacity penalty term
            # NEW: EMA prior metrics
            'ema_prior_mean_norm',  # L2 norm of EMA prior mean
            'ema_prior_std_mean',  # Mean of EMA prior standard deviations
            *[m.__name__ for m in metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('rec_loss', 'kl_loss', 'residual_loss', 'residual_rel_diff', *[m.__name__ for m in metric_ftns], writer=self.writer)

        self.data_key = config['trainer']['input_key']
        self.target_key = config['trainer']['output_key']
        self.input_const_keys = config['trainer'].get('input_const_keys', None)

        self.stablize_grad = config['trainer']['stablize_grad']
        self.stablize_count = 0

        # NEW (optional): running stats of u per-dim
        self.log_u_stats = True

        # NEW: Store initial learning rate for pretraining
        self.initial_lr = self.optimizer.param_groups[0]['lr']

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        
        # NEW: Reset learning rate to initial value during pretraining. #TODO: this function can be further cleaned up since many components are not used in the final version.
        if epoch < self.epochs_pretrain:
            # Pretraining phase: use initial learning rate (no scheduling)
            if self.optimizer.param_groups[0]['lr'] != self.initial_lr:
                self.optimizer.param_groups[0]['lr'] = self.initial_lr
                self.logger.info(f"Epoch {epoch}: Using initial LR for pretraining: {self.initial_lr}")
        else:
            # Training phase: let scheduler handle learning rate
            if epoch == self.epochs_pretrain:
                self.logger.info(f"Epoch {epoch}: Starting training phase, scheduler will control LR")
        
        sequence_len = None
        # Compute separate beta values for z_phy and z_aux when not in pretraining stage
        if not self.no_phy and epoch >= self.epochs_pretrain and self.use_kl_term_z_phy:
            # FIXED: Beta should start from 0 when training begins
            training_epoch = epoch - self.epochs_pretrain
            beta_z_phy = self.beta_max_z_phy * self._linear_annealing_epoch(training_epoch, warmup_epochs=self.beta_warmup)
        else:
            beta_z_phy = 0.0  # No KL loss during pretraining or when use_kl_term_z_phy=False
            
        if epoch >= self.epochs_pretrain and self.use_kl_term_z_aux:
            training_epoch = epoch - self.epochs_pretrain
            beta_z_aux = self.beta_max_z_aux * self._linear_annealing_epoch(training_epoch, warmup_epochs=self.beta_warmup)
        else:
            beta_z_aux = 0.0  # No KL loss during pretraining or when use_kl_term_z_aux=False

        # NEW: accumulators for u-stats
        u_sum = None
        u2_sum = None
        u_count = 0

        for batch_idx, data_dict in enumerate(self.data_loader):
            data = data_dict[self.data_key].to(self.device)
            input_const = {k: data_dict[k].to(self.device) for k in self.input_const_keys} if self.input_const_keys else None
            
            # Get time features if available
            time_feats = data_dict.get('time_feats', None)
            if time_feats is not None:
                time_feats = time_feats.to(self.device)
                if time_feats.dim() == 3:  # For sequence data
                    time_feats = time_feats.view(-1, time_feats.size(-1))

            if data.dim() == 3:
                # Sequence format: (batch_size, seq_len, features)
                sequence_len = data.size(1)
                data = data.view(-1, data.size(-1))

            self.optimizer.zero_grad()

            # Encode (u-space stats) with optional time features
            z_phy_stat, z_aux_stat = self.model.encode(data, time_feats)

            # NEW: Update EMA prior statistics (if enabled) - only after pretraining phase
            if self.use_ema_prior and not self.no_phy and epoch > 1 + self.epochs_pretrain:
                self.model.update_ema_prior(z_phy_stat['mean'], z_phy_stat['lnvar'])

            # Draw + decode
            # Use separate deterministic sampling for each latent variable based on their KL term settings
            hard_z_phy = not self.use_kl_term_z_phy  # Use deterministic sampling when KL term is disabled
            hard_z_aux = not self.use_kl_term_z_aux  # Use deterministic sampling when KL term is disabled
            z_phy, z_aux = self.model.draw(z_phy_stat, z_aux_stat, hard_z_phy=hard_z_phy, hard_z_aux=hard_z_aux)
            x_PB, x_P, y, delta, c = self.model.decode(z_phy, z_aux, epoch=epoch, epochs_pretrain=self.epochs_pretrain, full=True, const=input_const)

            # Losses - Get separate KL terms for flexible combination
            rec_loss, kl_u_phy, kl_z_aux = self._vae_loss(data, z_phy_stat, z_aux_stat, x_PB)
            
            # Calculate weighted KL loss
            kl_loss = beta_z_phy * kl_u_phy + beta_z_aux * kl_z_aux
            
            # ========================================================================
            # KL LOSS COMBINATION: Choose between original and capacity control modes
            # ========================================================================
            # Original mode: Combine KL terms first, then average (exactly as before)NOTE: effects of this term to be studied further
            # kl_loss = kl_u_phy + kl_z_aux

            # Compute L2 difference between raw physics output and corrected output
            residual_loss = torch.sum((x_PB - x_P).pow(2), dim=1).mean()
            # Also compute relative difference as percentage of raw output magnitude
            residual_rel_diff = torch.mean(torch.abs(x_PB - x_P) / (torch.abs(x_P) + 1e-8)) * 100.0

            # IMPROVED: Low-rank residual regularization terms
            if not self.no_phy and epoch >= self.epochs_pretrain and self.config['arch']['phys_vae']['dim_z_aux'] > 0:
                # Orthogonality penalty: λ_B ||B^T B - I||_F^2
                ortho_penalty = self.model.dec.orthogonality_penalty()
                
                # Coefficient penalty: λ_c ||c||_2^2
                coeff_penalty = torch.sum(c.pow(2), dim=1).mean()
                
                # Delta penalty: λ_Δ ||δ||_2^2
                delta_penalty = torch.sum(delta.pow(2), dim=1).mean()
            else:
                ortho_penalty = torch.tensor(0.0, device=data.device)
                coeff_penalty = torch.tensor(0.0, device=data.device)
                delta_penalty = torch.tensor(0.0, device=data.device)

            # NEW: Edge penalty for z_phy to avoid extreme values (0, 1)
            edge_penalty = self._edge_penalty(z_phy)
            
            # NEW: Temporal smoothness regularization for Mogi source parameters
            temporal_smoothness = self._temporal_smoothness_loss(z_phy, sequence_len)

            # Stage A: synthetic bootstrap (u-target)
            if not self.no_phy and epoch < self.epochs_pretrain:
                synthetic_data_loss = self._synthetic_data_loss(data.shape[0])
                loss = self.synthetic_data_loss_weight * synthetic_data_loss
                # Initialize capacity control variables for metrics
                capacity_penalty = torch.tensor(0.0, device=data.device)
                C_t = 0.0
            else:
                # ========================================================================
                # LOSS CALCULATION: Choose between original and capacity control modes
                # ========================================================================
                if self.use_capacity_control and not self.no_phy and epoch >= self.epochs_pretrain and self.use_kl_term_z_phy:
                    # --- CAPACITY CONTROL MODE ---
                    training_epoch = epoch - self.epochs_pretrain
                    warm = max(1, self.beta_warmup)  # your existing warmup (e.g., 50)
                    
                    C_t = min(self.C_max, self.C_max * training_epoch / warm)   # 0 -> C_max
                    capacity_penalty = self.C_gamma * (kl_u_phy - C_t)**2
                    
                    # Auxiliary KL with separate control
                    aux_term = beta_z_aux * kl_z_aux
                    
                    # Capacity control loss
                    loss = (rec_loss
                            + capacity_penalty
                            + aux_term
                            + self.ortho_penalty_weight * ortho_penalty
                            + self.coeff_penalty_weight * coeff_penalty
                            + self.delta_penalty_weight * delta_penalty
                            + self.edge_penalty_weight * edge_penalty
                            + self.temporal_smoothness_weight * temporal_smoothness)
                else:
                    # --- ORIGINAL MODE ---
                    # Original loss function with combined KL loss (rec_loss, ortho_penalty, and edge_penalty are the only terms that are used in the final version of PILA)
                    loss = (rec_loss + 
                           kl_loss +
                           self.ortho_penalty_weight * ortho_penalty +
                           self.coeff_penalty_weight * coeff_penalty +
                           self.delta_penalty_weight * delta_penalty +
                           self.edge_penalty_weight * edge_penalty +
                           self.temporal_smoothness_weight * temporal_smoothness)
                    
                    # Initialize capacity control variables for metrics (when not using capacity control)
                    C_t = 0.0
                    capacity_penalty = torch.tensor(0.0, device=data.device)

                # [REBUTTAL ONLY] physical-consistency term; inactive (weight 0)
                # for every config under configs/. Appended rather than folded
                # into the expressions above so those runs stay bit-identical.
                if self.syn_cotrain_weight > 0 and not self.no_phy:
                    syn_cotrain_loss = self._synthetic_consistency_loss_z(data.shape[0])
                    loss = loss + self.syn_cotrain_weight * syn_cotrain_loss
                    self.train_metrics.update('syn_data_loss', syn_cotrain_loss.item())

            loss.backward()

            # NEW: gradient clipping for stability
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            if self.stablize_grad:
                self._grad_stablizer(epoch, batch_idx, loss.item())

            self.optimizer.step()

            # Update metrics
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('rec_loss', rec_loss.item())
            self.train_metrics.update('kl_loss', kl_loss.item())
            self.train_metrics.update('residual_loss', residual_loss.item())
            self.train_metrics.update('residual_rel_diff', residual_rel_diff.item())
            
            self.train_metrics.update('kl_u_phy', kl_u_phy.item())
            self.train_metrics.update('kl_z_aux', kl_z_aux.item())
            self.train_metrics.update('beta_z_phy', beta_z_phy)
            self.train_metrics.update('beta_z_aux', beta_z_aux)
            
            self.train_metrics.update('capacity_target', C_t)
            self.train_metrics.update('capacity_penalty', capacity_penalty.item())
            
            # NEW: Track edge penalty
            self.train_metrics.update('edge_penalty', edge_penalty.item())
            
            # NEW: Track temporal smoothness
            self.train_metrics.update('temporal_smoothness', temporal_smoothness.item())
            
            # NEW: Track EMA prior metrics (only after pretraining phase when EMA starts updating)
            if self.use_ema_prior and not self.no_phy and hasattr(self.model, 'ema_mean') and epoch > 1 + self.epochs_pretrain:
                ema_mean_norm = torch.norm(self.model.ema_mean).item()
                ema_var = (self.model.ema_m2 - self.model.ema_mean**2).clamp(self.model.ema_min_var, self.model.ema_max_var)
                ema_std_mean = torch.mean(torch.sqrt(ema_var + 1e-8)).item()
                self.train_metrics.update('ema_prior_mean_norm', ema_mean_norm)
                self.train_metrics.update('ema_prior_std_mean', ema_std_mean)
            else:
                self.train_metrics.update('ema_prior_mean_norm', 0.0)
                self.train_metrics.update('ema_prior_std_mean', 1.0)
            
            if not self.no_phy and epoch < self.epochs_pretrain:
                self.train_metrics.update('syn_data_loss', synthetic_data_loss.item())
            else:
                # IMPROVED: Track all residual metrics
                self.train_metrics.update('ortho_penalty', ortho_penalty.item())
                self.train_metrics.update('coeff_penalty', coeff_penalty.item())
                self.train_metrics.update('delta_penalty', delta_penalty.item())
                
                # Track norms for monitoring
                c_norm = torch.norm(c, dim=1).mean().item() if c.numel() > 0 else 0.0
                delta_norm = torch.norm(delta, dim=1).mean().item()
                s_norm = torch.norm(self.model.dec.s).item() if hasattr(self.model.dec, 's') else 0.0
                basis_quality = torch.norm(torch.matmul(self.model.dec.B.T, self.model.dec.B) - torch.eye(self.model.dec.B.shape[1], device=self.model.dec.B.device), p='fro').item() if hasattr(self.model.dec, 'B') else 0.0
                
                self.train_metrics.update('c_norm', c_norm)
                self.train_metrics.update('delta_norm', delta_norm)
                self.train_metrics.update('s_norm', s_norm)
                self.train_metrics.update('basis_quality', basis_quality)

            # Optional u-stats
            if not self.no_phy and self.log_u_stats:
                u = z_phy_stat['mean'].detach()
                if u_sum is None:
                    u_sum = u.sum(dim=0)
                    u2_sum = (u**2).sum(dim=0)
                else:
                    u_sum += u.sum(dim=0)
                    u2_sum += (u**2).sum(dim=0)
                u_count += u.size(0)

            # Logging
            if batch_idx % self.config['trainer']['log_step'] == 0:
                if self.use_capacity_control:
                    kl_aux_str = f"KL_aux {kl_z_aux.item():.6f}" if self.dim_z_aux > 0 else "KL_aux 0.0"
                    log_str = (f"Train Ep {epoch} [{batch_idx}/{len(self.data_loader)}] "
                              f"Loss {loss.item():.6f} Rec {rec_loss.item():.6f} "
                              f"KL_u {kl_u_phy.item():.6f} {kl_aux_str} "
                              f"beta_z_phy {beta_z_phy:.3f} beta_z_aux {beta_z_aux:.3f} "
                              f"C_t {C_t:.3f} Cap_pen {capacity_penalty.item():.6f} "
                              f"Edge_pen {edge_penalty.item():.6f} "
                              f"Residual {residual_loss.item():.6f} "
                              f"residual_rel_diff {residual_rel_diff.item():.2f}%")
                else:
                    log_str = (f"Train Ep {epoch} [{batch_idx}/{len(self.data_loader)}] "
                              f"Loss {loss.item():.6f} Rec {rec_loss.item():.6f} "
                              f"KL(beta_z_phy={beta_z_phy:.3f},beta_z_aux={beta_z_aux:.3f}) {kl_loss.item():.6f} "
                              f"Edge_pen {edge_penalty.item():.6f} "
                              f"Residual {residual_loss.item():.6f} "
                              f"residual_rel_diff {residual_rel_diff.item():.2f}%")
                
                # Add EMA prior info to logging (only after pretraining phase when EMA starts updating)
                if self.use_ema_prior and not self.no_phy and hasattr(self.model, 'ema_mean') and epoch > 1 + self.epochs_pretrain:
                    ema_mean_norm = torch.norm(self.model.ema_mean).item()
                    ema_var = (self.model.ema_m2 - self.model.ema_mean**2).clamp(self.model.ema_min_var, self.model.ema_max_var)
                    ema_std_mean = torch.mean(torch.sqrt(ema_var + 1e-8)).item()
                    log_str += f" EMA_mean_norm {ema_mean_norm:.3f} EMA_std_mean {ema_std_mean:.3f}"
                
                if not self.no_phy and epoch >= self.epochs_pretrain:
                    log_str += f" Ortho {ortho_penalty.item():.6f} Coeff {coeff_penalty.item():.6f} Delta {delta_penalty.item():.6f}"
                
                # log_str += " (Gate: correction strength, Residual: physics vs corrected L2 diff)"
                self.logger.info(log_str)

        log = self.train_metrics.result()

        # Log epoch summary including residual_loss and current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        if self.use_capacity_control:
            kl_aux_summary = f"KL_aux: {log['kl_z_aux']:.6f}" if self.dim_z_aux > 0 else "KL_aux: 0.000000"
            summary_str = (f"Epoch {epoch} Summary - "
                          f"Loss: {log['loss']:.6f}, "
                          f"Rec: {log['rec_loss']:.6f}, "
                          f"KL_u: {log['kl_u_phy']:.6f}, "
                          f"{kl_aux_summary}, "
                          f"C_t: {log['capacity_target']:.3f}, "
                          f"Cap_pen: {log['capacity_penalty']:.6f}, "
                          f"Edge_pen: {log['edge_penalty']:.6f}, "
                          f"Residual: {log['residual_loss']:.6f}, "
                          f"residual_rel_diff: {log['residual_rel_diff']:.2f}%, "
                          f"LR: {current_lr:.6f}")
        else:
            summary_str = (f"Epoch {epoch} Summary - "
                          f"Loss: {log['loss']:.6f}, "
                          f"Rec: {log['rec_loss']:.6f}, "
                          f"KL: {log['kl_loss']:.6f}, "
                          f"Edge_pen: {log['edge_penalty']:.6f}, "
                          f"Residual: {log['residual_loss']:.6f}, "
                          f"residual_rel_diff: {log['residual_rel_diff']:.2f}%, "
                          f"LR: {current_lr:.6f}")
        
        # Add r(t) and tau monitoring (only when residual components exist)
        if not self.no_phy and self.dim_z_aux > 0:
            r_value = self.model.dec.get_r(epoch, self.epochs_pretrain)
            tau_value = self.model.dec.get_tau(epoch, self.epochs_pretrain)
            summary_str += f", r(t): {r_value:.3f}, tau: {tau_value:.3f}"
        
        # Add residual penalties if available
        if not self.no_phy and epoch >= self.epochs_pretrain:
            if 'ortho_penalty' in log:
                summary_str += f", Ortho: {log['ortho_penalty']:.6f}"
            if 'coeff_penalty' in log:
                summary_str += f", Coeff: {log['coeff_penalty']:.6f}"
            if 'delta_penalty' in log:
                summary_str += f", Delta: {log['delta_penalty']:.6f}"
        
        # Add EMA prior info to summary (only after pretraining phase when EMA starts updating)
        if self.use_ema_prior and not self.no_phy and 'ema_prior_mean_norm' in log and epoch > 1 + self.epochs_pretrain:
            summary_str += f", EMA_mean_norm: {log['ema_prior_mean_norm']:.3f}, EMA_std_mean: {log['ema_prior_std_mean']:.3f}"
        
        # summary_str += " (Gate: correction strength, Residual: physics vs corrected L2 diff)"
        self.logger.info(summary_str)

        # wandb logs
        wandb.log({f'train/{key}': value for key, value in log.items()})
        wandb.log({'train/lr': self.optimizer.param_groups[0]['lr']})
        wandb.log({'train/epoch': epoch})
        wandb.log({'train/beta_z_phy': beta_z_phy})
        wandb.log({'train/beta_z_aux': beta_z_aux})
        
        # Additional context for x_p_diff interpretation
        # if log['gate_mean'] > 0.5:
        #     wandb.log({'train/correction_comment': 'High correction activity (gate > 0.5)'})
        # else:
        #     wandb.log({'train/correction_comment': 'Low correction activity (gate < 0.5)'})

        # per-dim u stats (epoch-aggregated)
        if not self.no_phy and self.log_u_stats and u_count > 0:
            u_mean = (u_sum / u_count).cpu().numpy()
            u_var = (u2_sum / u_count).cpu().numpy() - u_mean**2
            u_std = np.sqrt(np.maximum(u_var, 1e-12))
            wandb.log({f'train/u_mean_dim{i}': float(u_mean[i]) for i in range(len(u_mean))})
            wandb.log({f'train/u_std_dim{i}': float(u_std[i]) for i in range(len(u_std))})

        # Validation
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            wandb.log({f'val/{key}': value for key, value in val_log.items()})
            wandb.log({'val/epoch': epoch})
            
            # Additional context for validation residual_loss interpretation
            if 'val_residual_rel_diff' in val_log:
                if val_log['val_residual_rel_diff'] > 10.0:
                    wandb.log({'val/correction_comment': 'High correction impact (>10% change)'})
                elif val_log['val_residual_rel_diff'] > 5.0:
                    wandb.log({'val/correction_comment': 'Moderate correction impact (5-10% change)'})
                else:
                    wandb.log({'val/correction_comment': 'Low correction impact (<5% change)'})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Override to save tau/r values for inference
        """
        arch = type(self.model).__name__
        
        # Get current tau/r values
        tau_r_values = None
        if not self.no_phy and self.dim_z_aux > 0:
            tau_r_values = self.model.dec.get_current_tau_r(epoch, self.epochs_pretrain)
        
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config,
            'tau_r_values': tau_r_values  # Save tau/r values for inference
        }
        
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, os.path.join(PARENT_DIR, filename))
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, os.path.join(PARENT_DIR, best_path))
            self.logger.info("Saving current best: model_best.pth ...")

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()

        total_rec_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, data_dict in enumerate(self.valid_data_loader):
                try:
                    data = data_dict[self.data_key].to(self.device)
                    input_const = {k: data_dict[k].to(self.device) for k in self.input_const_keys} if self.input_const_keys else None
                    
                    # Get time features if available
                    time_feats = data_dict.get('time_feats', None)
                    if time_feats is not None:
                        time_feats = time_feats.to(self.device)
                        if time_feats.dim() == 3:  # For sequence data
                            time_feats = time_feats.view(-1, time_feats.size(-1))
                    
                    if data.dim() == 3:
                        # Sequence format: (batch_size, seq_len, features)
                        sequence_len = data.size(1)
                        data = data.view(-1, data.size(-1))

                    # Get full model output to compute residual_loss
                    z_phy_stat, z_aux_stat = self.model.encode(data, time_feats)
                    # Use same hard_z settings as training
                    hard_z_phy = not self.use_kl_term_z_phy  # Use deterministic sampling when KL term is disabled
                    hard_z_aux = not self.use_kl_term_z_aux  # Use deterministic sampling when KL term is disabled
                    z_phy, z_aux = self.model.draw(z_phy_stat, z_aux_stat, hard_z_phy=hard_z_phy, hard_z_aux=hard_z_aux)
                    x_PB, x_P, y, delta, c = self.model.decode(z_phy, z_aux, epoch=epoch, epochs_pretrain=self.epochs_pretrain, full=True, const=input_const)
                    
                    # Use unified VAE loss function (same as training)
                    rec_loss, kl_u_phy, kl_z_aux = self._vae_loss(data, z_phy_stat, z_aux_stat, x_PB)
                    kl_loss = kl_u_phy + kl_z_aux # Original behavior: combine first, then average
                    
                    # Compute residual_loss for validation (L2 difference)
                    residual_loss = torch.sum((x_PB - x_P).pow(2), dim=1).mean()
                    # Also compute relative difference as percentage
                    residual_rel_diff = torch.mean(torch.abs(x_PB - x_P) / (torch.abs(x_P) + 1e-8)) * 100.0

                    self.valid_metrics.update('rec_loss', rec_loss.item())
                    self.valid_metrics.update('kl_loss', kl_loss.item())
                    self.valid_metrics.update('residual_loss', residual_loss.item())
                    self.valid_metrics.update('residual_rel_diff', residual_rel_diff.item())
                    
                    total_rec_loss += rec_loss.item()
                    total_kl_loss += kl_loss.item()
                    num_batches += 1

                except Exception as e:
                    self.logger.warning(f"Error in validation batch {batch_idx}: {e}")
                    continue

        avg_rec_loss = total_rec_loss / max(num_batches, 1)
        avg_kl_loss = total_kl_loss / max(num_batches, 1)
        
        # Compute average residual_loss for validation logging
        val_metrics = self.valid_metrics.result()
        avg_residual_loss = val_metrics['residual_loss']
        avg_residual_rel_diff = val_metrics['residual_rel_diff']
        self.logger.info(f"Validation Epoch: {epoch} Rec Loss: {avg_rec_loss:.6f} KL Loss: {avg_kl_loss:.6f} Residual: {avg_residual_loss:.6f} residual_rel_diff: {avg_residual_rel_diff:.2f}% (Physics vs Corrected)")
        return self.valid_metrics.result()

    def _vae_loss(self, data, z_phy_stat, z_aux_stat, x, pretrain=False):
        """
        VAE loss function that returns separate KL terms for flexible combination.
        Returns: rec_loss, kl_u_phy, kl_z_aux (separate terms for both modes)
        """
        # Use the configured loss function for reconstruction loss
        # rec_loss = torch.sum((x - data).pow(2), dim=1).mean()
        rec_loss = self.criterion(x, data) * self.rec_loss_scale

        n = data.shape[0]
        prior_u_phy_stat, prior_z_aux_stat = self.model.priors(n, self.device)

        # Physics KL in u-space (per-sample, not averaged yet)
        if not self.no_phy:
            KL_u_phy = kldiv_normal_normal(
                z_phy_stat['mean'], z_phy_stat['lnvar'],
                prior_u_phy_stat['mean'], prior_u_phy_stat['lnvar']
            ).mean()     # a scalar value, average of per-sample KL
        else:
            KL_u_phy = torch.zeros(n, device=self.device).mean()

        # Auxiliary KL (per-sample, not averaged yet)
        if pretrain or self.config['arch']['phys_vae']['dim_z_aux'] == 0:
            KL_z_aux = torch.zeros(n, device=self.device).mean()
        else:
            KL_z_aux = kldiv_normal_normal(
                z_aux_stat['mean'], z_aux_stat['lnvar'],
                prior_z_aux_stat['mean'], prior_z_aux_stat['lnvar']
            ).mean()     # a scalar value, average of per-sample KL

        # Return separate KLs for flexible combination
        return rec_loss, KL_u_phy, KL_z_aux


    def _synthetic_data_loss(self, batch_size):
        """
        Synthetic inversion loss in u-space:
        sample z~Uniform(0,1), simulate y, infer u_mean, and match to logit(z).
        
        This loss pretrains the encoder to learn the inverse mapping from physics
        outputs back to latent parameters, establishing a good initialization
        before introducing KL divergence terms.
        """
        if not self.no_phy:
            self.model.eval()
            with torch.no_grad():
                z = torch.rand((batch_size, self.dim_z_phy), device=self.device).clamp(1e-4, 1-1e-4)
                synthetic_y = self.model.generate_physonly(z)  # physics-only
            self.model.train()
            synthetic_features = self.model.enc.func_feat(synthetic_y)
            inferred_u_phy = self.model.enc.func_z_phy_mean(synthetic_features)  # u-mean
            target_u = torch.log(z) - torch.log1p(-z)  # logit(z)
            return torch.sum((inferred_u_phy - target_u).pow(2), dim=1).mean()
        else:
            return torch.zeros(1, device=self.device)

    def _synthetic_consistency_loss_z(self, batch_size):
        """
        [REBUTTAL ONLY] HVAE's physical-consistency loss, compared in z-space.

        Reviewer Q: "Could PILA be improved through addition of a physical
        consistency loss as in HVAE?"  This mirrors HVAE's
        PhysVAETrainer._synthetic_data_loss as closely as the two encoders allow:
        sample z ~ U(0,1), simulate the physics-only signal, re-infer the
        physical variables, and compare **in z-space**.

        The only difference from HVAE is where the squashing happens. HVAE's
        encoder head ends in Softplus and emits the physical value directly, so
        it compares its output against z as-is. PILA's head emits u and the
        sigmoid lives in `draw`, so the sigmoid is applied here to get the same
        quantity. Both therefore compare the drawn z against the sampled z.

        Not to be confused with _synthetic_data_loss above, which is the Stage-A
        bootstrap and regresses u against logit(z); that one is untouched and is
        what every published config would use (it is inactive at
        epochs_pretrain = 0).
        """
        if self.no_phy:
            return torch.zeros(1, device=self.device).squeeze()

        self.model.eval()
        with torch.no_grad():
            z = torch.rand((batch_size, self.dim_z_phy), device=self.device)
            synthetic_y = self.model.generate_physonly(z)  # physics-only signal
        self.model.train()
        synthetic_features = self.model.enc.func_feat(synthetic_y)
        # u -> z, the same map draw() applies, so this is the drawn z
        inferred_z_phy = torch.sigmoid(self.model.enc.func_z_phy_mean(synthetic_features))
        return torch.sum((inferred_z_phy - z).pow(2), dim=1).mean()

    def _edge_penalty(self, z_phy):
        """
        Edge penalty to encourage z_phy to avoid extreme values (0, 1).
        
        Args:
            z_phy: Physical variables tensor of shape (batch_size, dim_z_phy)
            
        Returns:
            Edge penalty: λ_edge * [-log(z) - log(1-z)]^power
        """
        if self.no_phy or z_phy is None:
            return torch.tensor(0.0, device=self.device)
        
        # Clamp z_phy to avoid numerical issues (fixed eps value)
        eps = 1e-6
        z_clamped = z_phy.clamp(eps, 1.0 - eps)
        
        # Compute edge penalty: -log(z) - log(1-z)
        edge_penalty = -torch.log(z_clamped) - torch.log(1.0 - z_clamped)
        
        # Apply power for fine-grained control
        edge_penalty = edge_penalty.pow(self.edge_penalty_power)
        
        # Return mean penalty across batch and dimensions
        return edge_penalty.mean()

    def _grad_stablizer(self, epoch, batch_idx, loss):
        para_grads = [v.grad.data for v in self.model.parameters(
        ) if v.grad is not None and torch.isnan(v.grad).any()]
        if len(para_grads) > 0:
            epsilon = 1e-7
            for v in para_grads:
                rand_values = torch.rand_like(v, dtype=torch.float)*epsilon
                mask = torch.isnan(v) | v.eq(0)
                v[mask] = rand_values[mask]
            self.stablize_count += 1
            self.logger.info(
                'epoch: {}, batch: {}, loss: {}, stablize count: {}'.format(
                    epoch, batch_idx, loss, self.stablize_count)
            )

    def _linear_annealing_epoch(self, epoch, warmup_epochs=30):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return 1.0

    def _temporal_smoothness_loss(self, z_phy, sequence_len):
        """
        Temporal smoothness regularization for Mogi source parameters.
        Encourages minimal variance in the spatial coordinates (xcen, ycen, d) 
        within each temporal sequence to enforce temporal smoothness.
        
        Args:
            z_phy: Physical variables tensor of shape (batch_size * seq_len, dim_z_phy)
            sequence_len: Length of temporal sequences (None if not using sequences)
            
        Returns:
            Temporal smoothness penalty: variance of spatial coordinates within sequences
        """
        if self.no_phy or z_phy is None or sequence_len is None:
            return torch.tensor(0.0, device=self.device)
        
        # Reshape z_phy back to sequence format: (batch_size, seq_len, dim_z_phy)
        batch_size = z_phy.size(0) // sequence_len
        if batch_size * sequence_len != z_phy.size(0):
            # If not evenly divisible, we can't apply temporal smoothness
            return torch.tensor(0.0, device=self.device)
        
        z_phy_seq = z_phy.view(batch_size, sequence_len, -1)  # (batch_size, seq_len, dim_z_phy)
        
        # For Mogi model, dim_z_phy=4: [xcen, ycen, d, dV]
        # We apply smoothness to the first 3 dimensions (spatial coordinates)
        spatial_coords = z_phy_seq[:, :, :3]  # (batch_size, seq_len, 3)
        
        # Compute variance across the sequence dimension for each batch
        # var across seq_len dimension: (batch_size, 3)
        coord_variance = torch.var(spatial_coords, dim=1)  # (batch_size, 3)
        
        # Return mean variance across all batches and spatial dimensions
        return coord_variance.mean()
