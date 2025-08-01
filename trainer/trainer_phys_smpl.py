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

        # Flag for physics model usage
        self.no_phy = config['arch']['phys_vae']['no_phy'] # No physics model, default False

        # Pretraining epochs
        self.epochs_pretrain = config['trainer']['phys_vae']['epochs_pretrain'] # default 0

        # Number of physical variables
        self.dim_z_phy = config['arch']['phys_vae']['dim_z_phy'] #7 for RTM, 4 for Mogi

        # Phys-VAE specific configurations
        self.kl_loss_weight = config['trainer']['phys_vae']['balance_kld'] #NOTE modify the config file accordingly
        self.least_action_loss_weight = config['trainer']['phys_vae']['balance_lact']
        self.synthetic_data_loss_weight = config['trainer']['phys_vae']['balance_data_aug']
    
        # Metric tracker
        # self.train_metrics = MetricTracker(
        #     'loss', 'rec_loss', 'kl_loss', 'unmix_loss', 'syn_data_loss', 'least_act_loss', 'smoothness_loss',
        #     *[m.__name__ for m in metric_ftns], writer=self.writer)
        self.train_metrics = MetricTracker(
            'loss', 'rec_loss', 'kl_loss', 'least_act_loss', 'syn_data_loss', 'smoothness_loss',
            *[m.__name__ for m in metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker(
            'rec_loss', 'kl_loss',
            *[m.__name__ for m in metric_ftns], writer=self.writer)# loss here refers to rec_loss

        # define the data key and target key
        self.data_key = config['trainer']['input_key']
        self.target_key = config['trainer']['output_key']
        # define the input-specific constants to be used in the physic model
        if 'input_const_keys' in config['trainer']:
            self.input_const_keys = config['trainer']['input_const_keys']
        else:   
            self.input_const_keys = None
        # define a flag to indicate whether to stablize the gradient or not
        self.stablize_grad = config['trainer']['stablize_grad']
        self.stablize_count = 0

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch.

        :param epoch: Current epoch number.
        :return: A dictionary with metrics for the epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        
        # Sequence length, for the 'GPSSeqDataLoader' used in Mogi model
        seqence_len = None
        annealing_weight = self._cylindrical_annealing_epoch(epoch-1)#the epoch starts with 1, but it ought to start with 0 for annealing

        for batch_idx, data_dict in enumerate(self.data_loader):
            data = data_dict[self.data_key].to(self.device)
            # Load the angle specific information here
            if self.input_const_keys is not None:
                input_const = {k: data_dict[k].to(self.device) for k in self.input_const_keys}
            else:
                input_const = None

            if data.dim() == 3:
                seqence_len = data.size(1)
                data = data.view(-1, data.size(-1))

            self.optimizer.zero_grad()

            # Encode step: Infer latent variables
            z_phy_stat, z_aux_stat, unmixed = self.model.encode(data)

            # Draw step: Sample latent variables
            z_phy, z_aux = self.model.draw(z_phy_stat, z_aux_stat, hard_z=False)
            # z_phy, z_aux = self.model.draw(z_phy_stat, z_aux_stat, hard_z=True)

            # Decode step: Reconstruct outputs
            x_PB, x_P, y = self.model.decode(z_phy, z_aux, full=True, const = input_const)

            # ELBO loss
            rec_loss, kl_loss = self._vae_loss(data, z_phy_stat, z_aux_stat, x_PB)

            # Unmixing regularization (R_{unmix})
            unmix_loss = self._unmixing_loss(unmixed, y)

            # Synthetic data regularization (R_{DA,2})
            synthetic_data_loss = self._synthetic_data_loss(data.shape[0])

            # Sequence smoothness regularization (for Mogi model) #NOTE used only in Mogi model
            smoothness_loss = self._sequence_smoothness_loss(x_P, seqence_len)

            # Loss calculations
            # ELBO loss #NOTE here it's sample-wise loss, not element-wise loss
            if not self.no_phy and epoch < self.epochs_pretrain:
                # rec_loss, kl_loss = self._vae_loss(data, z_phy_stat, z_aux_stat, x_P, pretrain=True)
                # least_action_loss = torch.zeros(1, device=self.device)
                # loss = rec_loss \
                #        + annealing_weight * kl_loss \
                #        + 0.01 * smoothness_loss  # NOTE no least action loss in pretraining phase
                loss = self.synthetic_data_loss_weight * synthetic_data_loss 
                
            else:
                rec_loss, kl_loss = self._vae_loss(data, z_phy_stat, z_aux_stat, x_PB)
                
                # Synthetic data regularization (R_{DA,2}) #NOTE this part can be ablated
                # synthetic_data_loss = self._synthetic_data_loss(data.shape[0])

                # Least action regularization (R_{ppc}) 
                least_action_loss = self._least_action_loss(x_PB, x_P) 
                # Compute the loss
                # loss = rec_loss \
                #     + annealing_weight * kl_loss \
                #     + 0.1 * least_action_loss \
                #     + 0.01 * smoothness_loss

                # loss = rec_loss\
                #       + 0.001 * kl_loss \
                #       + self.least_action_loss_weight * least_action_loss \
                #       + self.synthetic_data_loss_weight * synthetic_data_loss 


                # NOTE loss to experiment on the full version of PhysVAE
                # loss = rec_loss \
                #     + self._loss_weight(rec_loss, kl_loss) * kl_loss \
                #     + self._loss_weight(rec_loss, unmix_loss) * unmix_loss \
                #     + self._loss_weight(rec_loss, synthetic_data_loss) * synthetic_data_loss \
                #     + self._loss_weight(rec_loss, least_action_loss) * least_action_loss \
                #     + 0.01 * smoothness_loss
                
                loss = rec_loss \
                    + self.kl_loss_weight * kl_loss * torch.exp(-9) \
                    + self._loss_weight(rec_loss, unmix_loss) * unmix_loss \
                    + self._loss_weight(rec_loss, synthetic_data_loss) * synthetic_data_loss \
                    + self._loss_weight(rec_loss, least_action_loss) * least_action_loss \
                    + 0.01 * smoothness_loss
                
            # Backpropagation
            loss.backward()

            # gradient stabilisation

            if self.stablize_grad:
                self._grad_stablizer(epoch, batch_idx, loss.item())

            self.optimizer.step()

            # Update metrics
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('rec_loss', rec_loss.item())
            self.train_metrics.update('kl_loss', kl_loss.item())
            self.train_metrics.update('unmix_loss', unmix_loss.item())#NOTE ablation on unmix_loss first
            self.train_metrics.update('syn_data_loss', synthetic_data_loss.item())
            self.train_metrics.update('least_act_loss', least_action_loss.item())
            self.train_metrics.update('smoothness_loss', smoothness_loss.item())

            # Logging
            if batch_idx % self.config['trainer']['log_step'] == 0:
                self.logger.info(f"Train Epoch: {epoch} [{batch_idx}/{len(self.data_loader)}] "
                                 f"Loss: {loss.item():.6f} Rec Loss: {rec_loss.item():.6f} "
                                 f"KL Loss: {kl_loss.item():.6f} "
                                 f"Unmix Loss: {unmix_loss.item():.6f} "#NOTE ablation on unmix_loss first
                                 f"Synthetic Data Loss: {synthetic_data_loss.item():.6f} "
                                 f"Least Action Loss: {least_action_loss.item():.6f} "
                                 f"Smoothness Loss: {smoothness_loss.item():.6f}")

        log = self.train_metrics.result()

        # Log the training metrics to wandb
        wandb.log({f'train/{key}': value for key, value in log.items()})
        wandb.log({'train/lr': self.optimizer.param_groups[0]['lr']})
        wandb.log({'train/epoch': epoch})
        # wandb.log({'train/annealing_weight': annealing_weight})

        # Validation
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            # Log the validation metrics to wandb
            wandb.log({f'val/{key}': value for key, value in val_log.items()})
            wandb.log({'val/epoch': epoch})

        # Learning rate scheduler
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validation logic after training an epoch.

        :param epoch: Current epoch number.
        :return: A dictionary with validation metrics.
        """
        self.model.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            for batch_idx, data_dict in enumerate(self.valid_data_loader):
                data = data_dict[self.data_key].to(self.device)
                # target = data_dict[self.target_key].to(self.device)
                if self.input_const_keys is not None:
                    input_const = {k: data_dict[k].to(self.device) for k in self.input_const_keys}
                else:
                    input_const = None
                if data.dim() == 3:
                    data = data.view(-1, data.size(-1))
                    
                data = data.to(self.device)

                z_phy_stat, z_aux_stat, x = self.model(data, hard_z=False, const = input_const)
                # z_phy_stat, z_aux_stat, x = self.model(data, hard_z=True, const = input_const)

                rec_loss, kl_loss = self._vae_loss(data, z_phy_stat, z_aux_stat, x)

                # Update metrics 
                self.valid_metrics.update('rec_loss', rec_loss.item())
                self.valid_metrics.update('kl_loss', kl_loss.item())

        # Log the validation metrics
        self.logger.info(f"Validation Epoch: {epoch} Rec Loss: {rec_loss.item():.6f} KL Loss: {kl_loss.item():.6f}")

        return self.valid_metrics.result()

    def _vae_loss(self, data, z_phy_stat, z_aux_stat, x, pretrain=False):
        """
        Compute the VAE loss for the model.

        :param data: Input data.
        :param z_phy_stat: Dictionary with mean and log-variance for z_phy.
        :param z_aux_stat: Dictionary with mean and log-variance for z_aux.
        :param x: Reconstruction of the input data.
        :return: Reconstruction loss and KL divergence loss.
        """
        n = data.shape[0]
        # rec_loss = mse_loss(x, data)#NOTE this is MSE can be experimented at the end, but for now keep the same implementation as PhysVAE
        rec_loss = torch.sum((x-data).pow(2), dim=1).mean()
        prior_z_phy_stat, prior_z_aux_stat = self.model.priors(n, self.device) #TODO

        KL_z_phy = kldiv_normal_normal(z_phy_stat['mean'], z_phy_stat['lnvar'],
            prior_z_phy_stat['mean'], prior_z_phy_stat['lnvar']
            ) if not self.no_phy else torch.zeros(1, device=self.device)
        # KL_z_phy = torch.zeros(1, device=self.device)
        
        if pretrain:
            KL_z_aux = torch.zeros(1, device=self.device)
            # kl_loss = torch.zeros(1, device=self.device)
        else:
            KL_z_aux = kldiv_normal_normal(z_aux_stat['mean'], z_aux_stat['lnvar'],
                prior_z_aux_stat['mean'], prior_z_aux_stat['lnvar']
                ) if self.config['arch']['phys_vae']['dim_z_aux'] > 0 else torch.zeros(1, device=self.device)
        
        kl_loss = (KL_z_phy + KL_z_aux).mean()
        # kl_loss = torch.zeros(1, device=self.device)

        return rec_loss, kl_loss
    
    def _unmixing_loss(self, unmixed, y):
        """
        Compute the unmixing regularization loss.

        :param unmixed: Physics-unmixed signal.
        :param y: Physics-only component generated by the physics model.
        :return: Unmixing loss.
        """
        if not self.no_phy:
            return torch.sum((unmixed - y.detach()).pow(2), dim=1).mean()
        else:
            return torch.zeros(1, device=self.device)
        
    def _synthetic_data_loss(self, batch_size):
        """
        Generate synthetic data and compute loss for regularization.

        :param batch_size: Number of samples in the current batch.
        :return: Synthetic data loss.
        """
        if not self.no_phy:
            self.model.eval()
            with torch.no_grad():
                synthetic_z_phy = torch.rand((batch_size, self.dim_z_phy), device=self.device) # Generate synthetic data in the unit scale
                synthetic_y = self.model.generate_physonly(synthetic_z_phy)  # Simulated physics-only signal
            self.model.train()
            synthetic_features = self.model.enc.func_feat(synthetic_y)
            inferred_z_phy = self.model.enc.func_z_phy_mean(synthetic_features)
            return torch.sum((inferred_z_phy - synthetic_z_phy).pow(2), dim=1).mean()  # Regularization term
        else:
            return torch.zeros(1, device=self.device)

    def _least_action_loss(self, x_PB, x_P):
        """
        Compute the least action regularization loss.

        :param x_PB: Predicted physics-based signal.
        :param x_P: Predicted physics-only signal.
        :return: Least action loss.
        """
        #NOTE TODO it can also be ridge regression loss, to be experimented at the end
        if not self.no_phy:
            return torch.sum((x_PB - x_P).pow(2), dim=1).mean()
            # return mse_loss(x_PB, x_P)#NOTE this is MSE can be experimented at the end
        else:
            return torch.zeros(1, device=self.device)
    
    def _sequence_smoothness_loss(self, x_P, seqence_len=None):
        """
        Compute the smoothness regularization loss for a sequence.

        :param x_P: Predicted physics-only signal.
        :param seqence_len: Length of the sequence.
        :return: Smoothness loss.
        """
        if not self.no_phy and seqence_len is not None:
            x_P_reshaped = x_P.view(-1, seqence_len, x_P.size(-1))
            diff = x_P_reshaped[:, 1:, :] - x_P_reshaped[:, :-1, :]
            return torch.mean(diff**2)
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

    def _loss_weight(self, rec_loss, reg_loss):
        """
        Compute the weighted loss so that the reg_loss is always one order of magnitude smaller than rec_loss.

        :param rec_loss: Reconstruction loss.
        :param reg_loss: Regularization loss.
        :return: Weighted loss.
        """
        # Compute the magnitude of the losses
        rec_loss_mag = 10 ** torch.floor(torch.log10(rec_loss + 1e-6))
        reg_loss_mag = 10 ** torch.floor(torch.log10(reg_loss + 1e-6))
        weight = rec_loss_mag / max(reg_loss_mag, 1e-6)
        return weight.detach()/10 # One order of magnitude smaller
    
    def _linear_annealing_epoch(self, epoch, warmup_epochs=30):
        """
        Linear annealing of the loss weight based on the epoch number.

        :param epoch: Current epoch number.
        :param warmup_epochs: Number of epochs for warmup.
        :return: Annealed loss weight.
        """
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return 1.0
        
    def _cylindrical_annealing_epoch(self, epoch: int,
                                cycle_length: int = 10,
                                ramp_frac: float = 0.5) -> float:
        """
        Compute a cylindrical (cyclical + flat‐top) annealing factor per‐epoch.

        Parameters:
            epoch         – Current epoch index (0‐based).
            cycle_length  – Number of epochs in one full cycle.
            ramp_frac     – Fraction of each cycle spent ramping (0→1).
                            The remaining (1−ramp_frac) is flat at β=1.

        Returns:
            beta          – Annealing weight in [0,1].
        """
        # Position within the cycle
        pos = epoch % cycle_length
        # Number of epochs spent ramping up
        ramp_epochs = int(cycle_length * ramp_frac)
        if ramp_epochs <= 0:
            return 1.0
        # Linear ramp up
        beta = pos / ramp_epochs
        # Clamp at 1.0
        return min(beta, 1.0)
