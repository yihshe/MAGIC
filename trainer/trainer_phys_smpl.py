"""
Simplified version of PhysVAE for ablation study.
TODO add wandb logging to check the balance between different terms.
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

        # NEW: gradient clipping
        self.grad_clip_norm = config['trainer'].get('grad_clip_norm', 1.0)

        # trackers
        self.train_metrics = MetricTracker(
            'loss', 'rec_loss', 'kl_loss', 'gate_mean',
            'syn_data_loss',  # diagnostic
            *[m.__name__ for m in metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('rec_loss', 'kl_loss', *[m.__name__ for m in metric_ftns], writer=self.writer)

        self.data_key = config['trainer']['input_key']
        self.target_key = config['trainer']['output_key']
        self.input_const_keys = config['trainer'].get('input_const_keys', None)

        self.stablize_grad = config['trainer']['stablize_grad']
        self.stablize_count = 0

        # NEW (optional): running stats of u per-dim
        self.log_u_stats = True

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        
        seqence_len = None
        beta = self._linear_annealing_epoch(epoch-1, warmup_epochs=self.beta_warmup)  # NEW

        # NEW: accumulators for u-stats
        u_sum = None
        u2_sum = None
        u_count = 0

        for batch_idx, data_dict in enumerate(self.data_loader):
            data = data_dict[self.data_key].to(self.device)
            input_const = {k: data_dict[k].to(self.device) for k in self.input_const_keys} if self.input_const_keys else None

            if data.dim() == 3:
                seqence_len = data.size(1)
                data = data.view(-1, data.size(-1))

            self.optimizer.zero_grad()

            # Encode (u-space stats)
            z_phy_stat, z_aux_stat = self.model.encode(data)

            # Draw + decode
            z_phy, z_aux = self.model.draw(z_phy_stat, z_aux_stat, hard_z=False)
            x_PB, x_P, y, gate = self.model.decode(z_phy, z_aux, full=True, const=input_const)

            # Losses
            rec_loss, kl_loss = self._vae_loss(data, z_phy_stat, z_aux_stat, x_PB)
            gate_mean = gate.mean()

            # Stage A: synthetic bootstrap (u-target)
            if not self.no_phy and epoch < self.epochs_pretrain:
                synthetic_data_loss = self._synthetic_data_loss(data.shape[0])
                loss = self.synthetic_data_loss_weight * synthetic_data_loss
            else:
                loss = rec_loss + beta * kl_loss + self.gate_loss_weight * gate_mean

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
            self.train_metrics.update('gate_mean', gate_mean.item())
            if not self.no_phy and epoch < self.epochs_pretrain:
                self.train_metrics.update('syn_data_loss', synthetic_data_loss.item())

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
                self.logger.info(
                    f"Train Ep {epoch} [{batch_idx}/{len(self.data_loader)}] "
                    f"Loss {loss.item():.6f} Rec {rec_loss.item():.6f} "
                    f"KL(beta={beta:.3f}) {kl_loss.item():.6f} "
                    f"Gate(mean) {gate_mean.item():.6f}"
                )

        log = self.train_metrics.result()

        # wandb logs
        wandb.log({f'train/{key}': value for key, value in log.items()})
        wandb.log({'train/lr': self.optimizer.param_groups[0]['lr']})
        wandb.log({'train/epoch': epoch})
        wandb.log({'train/beta': beta})

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

                    z_phy_stat, z_aux_stat, x = self.model(data, hard_z=False, const=input_const)  # returns x_mean
                    rec_loss, kl_loss = self._vae_loss(data, z_phy_stat, z_aux_stat, x)

                    self.valid_metrics.update('rec_loss', rec_loss.item())
                    self.valid_metrics.update('kl_loss', kl_loss.item())
                    
                    total_rec_loss += rec_loss.item()
                    total_kl_loss += kl_loss.item()
                    num_batches += 1

                except Exception as e:
                    self.logger.warning(f"Error in validation batch {batch_idx}: {e}")
                    continue

        avg_rec_loss = total_rec_loss / max(num_batches, 1)
        avg_kl_loss = total_kl_loss / max(num_batches, 1)
        
        self.logger.info(f"Validation Epoch: {epoch} Rec Loss: {avg_rec_loss:.6f} KL Loss: {avg_kl_loss:.6f}")
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
