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
from torchvision.utils import make_grid
from base import BaseTrainer
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

        # CHANGED: minimal knobs
        self.beta_warmup = config['trainer']['phys_vae'].get('kl_warmup_epochs', 50)  # NEW
        self.gate_loss_weight = config['trainer']['phys_vae'].get('balance_gate', 1e-3)  # NEW

        # for Stage A bootstrap in u-space
        self.synthetic_data_loss_weight = config['trainer']['phys_vae'].get('balance_data_aug', 1.0)
        
        # NEW: Loss weights for improved residual architecture
        self.ortho_penalty_weight = config['trainer']['phys_vae'].get('ortho_penalty_weight', 0.1)
        self.coeff_penalty_weight = config['trainer']['phys_vae'].get('coeff_penalty_weight', 1e-4)
        self.delta_penalty_weight = config['trainer']['phys_vae'].get('delta_penalty_weight', 1e-4)

        # NEW: gradient clipping
        self.grad_clip_norm = config['trainer'].get('grad_clip_norm', 1.0)

        # trackers
        self.train_metrics = MetricTracker(
            'loss', 'rec_loss', 'kl_loss',
            'syn_data_loss',  # diagnostic
            'residual_loss',  # L2 difference between raw and corrected output
            'residual_rel_diff',  # relative difference as percentage
            'ortho_penalty',  # orthogonality penalty for basis matrix
            'coeff_penalty',  # coefficient L2 penalty
            'delta_penalty',  # delta L2 penalty
            'c_norm',  # norm of coefficient vector
            'delta_norm',  # norm of residual vector
            's_norm',  # norm of scale parameters
            'basis_quality',  # ||B^T B - I||_F
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
        
        # NEW: Reset learning rate to initial value during pretraining
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
        # Only compute beta when not in pretraining stage
        if not self.no_phy and epoch >= self.epochs_pretrain:
            # FIXED: Beta should start from 0 when training begins
            training_epoch = epoch - self.epochs_pretrain
            beta = self._linear_annealing_epoch(training_epoch, warmup_epochs=self.beta_warmup)
        else:
            beta = 0.0  # No KL loss during pretraining

        # NEW: accumulators for u-stats
        u_sum = None
        u2_sum = None
        u_count = 0

        for batch_idx, data_dict in enumerate(self.data_loader):
            data = data_dict[self.data_key].to(self.device)
            input_const = {k: data_dict[k].to(self.device) for k in self.input_const_keys} if self.input_const_keys else None

            if data.dim() == 3:
                sequence_len = data.size(1)
                data = data.view(-1, data.size(-1))

            self.optimizer.zero_grad()

            # Encode (u-space stats)
            z_phy_stat, z_aux_stat = self.model.encode(data)

            # Draw + decode
            z_phy, z_aux = self.model.draw(z_phy_stat, z_aux_stat, hard_z=False)
            x_PB, x_P, y, delta, c = self.model.decode(z_phy, z_aux, epoch=epoch, epochs_pretrain=self.epochs_pretrain, full=True, const=input_const)

            # Losses
            rec_loss, kl_loss = self._vae_loss(data, z_phy_stat, z_aux_stat, x_PB)

            # Compute L2 difference between raw physics output and corrected output
            residual_loss = torch.sum((x_PB - x_P).pow(2), dim=1).mean()
            # Also compute relative difference as percentage of raw output magnitude
            residual_rel_diff = torch.mean(torch.abs(x_PB - x_P) / (torch.abs(x_P) + 1e-8)) * 100.0

            # IMPROVED: Low-rank residual regularization terms
            if not self.no_phy and epoch >= self.epochs_pretrain:
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

            # Stage A: synthetic bootstrap (u-target)
            if not self.no_phy and epoch < self.epochs_pretrain:
                synthetic_data_loss = self._synthetic_data_loss(data.shape[0])
                loss = self.synthetic_data_loss_weight * synthetic_data_loss
            else:
                # IMPROVED: Complete loss function
                loss = (rec_loss + beta * kl_loss + 
                       self.ortho_penalty_weight * ortho_penalty +
                       self.coeff_penalty_weight * coeff_penalty +
                       self.delta_penalty_weight * delta_penalty)

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
                basis_quality = torch.norm(torch.matmul(self.model.dec.B.T, self.model.dec.B) - torch.eye(self.model.dec.B.shape[1], device=self.model.dec.B.device), p='fro').item()
                
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
                log_str = (f"Train Ep {epoch} [{batch_idx}/{len(self.data_loader)}] "
                          f"Loss {loss.item():.6f} Rec {rec_loss.item():.6f} "
                          f"KL(beta={beta:.3f}) {kl_loss.item():.6f} "
                          f"Residual {residual_loss.item():.6f} "
                          f"residual_rel_diff {residual_rel_diff.item():.2f}%")
                
                if not self.no_phy and epoch >= self.epochs_pretrain:
                    log_str += f" Ortho {ortho_penalty.item():.6f} Coeff {coeff_penalty.item():.6f} Delta {delta_penalty.item():.6f}"
                
                # log_str += " (Gate: correction strength, Residual: physics vs corrected L2 diff)"
                self.logger.info(log_str)

        log = self.train_metrics.result()

        # Log epoch summary including residual_loss and current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        summary_str = (f"Epoch {epoch} Summary - "
                      f"Loss: {log['loss']:.6f}, "
                      f"Rec: {log['rec_loss']:.6f}, "
                      f"KL: {log['kl_loss']:.6f}, "
                      f"Residual: {log['residual_loss']:.6f}, "
                      f"residual_rel_diff: {log['residual_rel_diff']:.2f}%, "
                      f"LR: {current_lr:.6f}")
        
        # Add r(t) and tau monitoring
        if not self.no_phy:
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
        
        # summary_str += " (Gate: correction strength, Residual: physics vs corrected L2 diff)"
        self.logger.info(summary_str)

        # wandb logs
        wandb.log({f'train/{key}': value for key, value in log.items()})
        wandb.log({'train/lr': self.optimizer.param_groups[0]['lr']})
        wandb.log({'train/epoch': epoch})
        wandb.log({'train/beta': beta})
        
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
                    if data.dim() == 3:
                        data = data.view(-1, data.size(-1))

                    # Get full model output to compute residual_loss
                    z_phy_stat, z_aux_stat = self.model.encode(data)
                    z_phy, z_aux = self.model.draw(z_phy_stat, z_aux_stat, hard_z=False)
                    x_PB, x_P, y, delta, c = self.model.decode(z_phy, z_aux, epoch=epoch, epochs_pretrain=self.epochs_pretrain, full=True, const=input_const)
                    
                    rec_loss, kl_loss = self._vae_loss(data, z_phy_stat, z_aux_stat, x_PB)
                    
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
        CHANGED: KL computed in u-space; auxiliaries unchanged.
        """
        rec_loss = torch.sum((x - data).pow(2), dim=1).mean()

        n = data.shape[0]
        prior_u_phy_stat, prior_z_aux_stat = self.model.priors(n, self.device)

        KL_u_phy = kldiv_normal_normal(z_phy_stat['mean'], z_phy_stat['lnvar'],
                                       prior_u_phy_stat['mean'], prior_u_phy_stat['lnvar']) \
                   if not self.no_phy else torch.zeros(1, device=self.device)

        if pretrain:
            KL_z_aux = torch.zeros(1, device=self.device)
        else:
            KL_z_aux = kldiv_normal_normal(z_aux_stat['mean'], z_aux_stat['lnvar'],
                                           prior_z_aux_stat['mean'], prior_z_aux_stat['lnvar']) \
                       if self.config['arch']['phys_vae']['dim_z_aux'] > 0 else torch.zeros(1, device=self.device)
        
        kl_loss = (KL_u_phy + KL_z_aux).mean()
        return rec_loss, kl_loss

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
