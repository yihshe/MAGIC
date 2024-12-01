# Adapted from the training script of Phys-VAE
import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, kldiv_normal_normal
import wandb
from model.loss import mse_loss, mse_loss_per_channel
from IPython import embed

class PhysVAETrainer(BaseTrainer):
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
        # TODO check the weight used for KL divergence in general: 1) weight extremely small, same as AE? 
        # 2) we have already applied a physical range as a prior, what we learn is a unit vector
        # 3) the physical model is determinstic, then whether variational is necessary or not.
        self.kl_loss_weight = config['trainer']['phys_vae']['balance_kld'] + config['trainer']['phys_vae']['balance_lact_enc'] 
        self.unmix_loss_weight = config['trainer']['phys_vae']['balance_unmix']
        self.synthetic_data_loss_weight = config['trainer']['phys_vae']['balance_data_aug']
        self.least_action_loss_weight = config['trainer']['phys_vae']['balance_lact_dec']
        
        # Metric tracker
        self.train_metrics = MetricTracker(
            'loss', 'rec_loss', 'kl_loss', 'unmix_loss', 'physics_loss', 'synthetic_data_loss',
            *[m.__name__ for m in metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker(
            'loss', 'rec_loss', 'kl_loss', 'unmix_loss', 'physics_loss', 'synthetic_data_loss',
            *[m.__name__ for m in metric_ftns], writer=self.writer)

        # define the data key and target key
        self.data_key = config['trainer']['input_key']
        self.target_key = config['trainer']['output_key']
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

        for batch_idx, (data,) in enumerate(self.data_loader):
            # TODO data and target for the data structure
            # data = data_dict[self.data_key].to(self.device)
            # target = data_dict[self.target_key].to(self.device)
            data = data.to(self.device)
            self.optimizer.zero_grad()

            # Encode step: Infer latent variables
            z_phy_stat, z_aux2_stat, unmixed = self.model.encode(data)

            # Draw step: Sample latent variables
            z_phy, z_aux2 = self.model.draw(z_phy_stat, z_aux2_stat, hard_z=False)

            # Decode step: Reconstruct outputs
            x_PB, x_P, x_lnvar, y = self.model.decode(z_phy, z_aux2, full=True)

            # Reconstruction variance
            x_var = torch.exp(x_lnvar)

            # Loss calculations
            # TODO whether data and x_PB need to be rescaled 
            # ELBO loss
            rec_loss, kl_loss = self._vae_loss(data, z_phy_stat, z_aux2_stat, x_PB)
            
            # Unmixing regularization (R_{DA,1})
            unmix_loss = self._unmixing_loss(unmixed, y)  # Unmixing regularization TODO reg_unmix (R_{DA,1})

            # Synthetic data regularization (R_{DA,2})
            synthetic_data_loss = self._synthetic_data_loss(data.shape[0])

            # Least action regularization (R_{ppc}) NOTE: until here
            least_action_loss = self._least_action_loss(x_PB, x_P) 

            
            # Total loss 
            if not self.no_phy and epoch < self.epochs_pretrain: 
                # Pretraining phase, not used in default setting
                loss = self.synthetic_data_loss_weight * synthetic_data_loss
            else:
                # Training phase TODO loss weights TBD
                loss = rec_loss \
                    + self.kl_loss_weight * kl_loss * x_var.detach() \
                    + self.unmix_loss_weight * unmix_loss \
                    + self.synthetic_data_loss_weight * synthetic_data_loss \
                    + self.least_action_loss_weight * least_action_loss
            
            # Backpropagation
            loss.backward()

            # TODO gradient clipping or stabilisation
            # if args.grad_clip>0.0:
            # torch.nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
            if self.stablize_grad:
                self._grad_stablizer(epoch, batch_idx, loss.item())

            self.optimizer.step()

            # Update metrics TODO double-check this metrics
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('rec_loss', rec_loss.item())
            self.train_metrics.update('kl_loss', kl_loss.item())
            self.train_metrics.update('unmix_loss', unmix_loss.item())
            self.train_metrics.update('syn_data_loss', synthetic_data_loss.item())
            self.train_metrics.update('least_act_loss', least_action_loss.item())

            # Logging
            if batch_idx % self.config['trainer']['log_step'] == 0:
                self.logger.info(f"Train Epoch: {epoch} [{batch_idx}/{len(self.data_loader)}] "
                                 f"Loss: {loss.item():.6f} Rec Loss: {rec_loss.item():.6f} "
                                 f"KL Loss: {kl_loss.item():.6f} Unmix Loss: {unmix_loss.item():.6f}"
                                 f"Synthetic Data Loss: {synthetic_data_loss.item():.6f} "
                                 f"Least Action Loss: {least_action_loss.item():.6f}")

        log = self.train_metrics.result()

        # Validation
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

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
            for batch_idx, (data,) in enumerate(self.valid_data_loader):
                # TODO data and target for the data structure
                # data = data_dict[self.data_key].to(self.device)
                # target = data_dict[self.target_key].to(self.device)
                data = data.to(self.device)

                z_phy_stat, z_aux2_stat, x, _ = self.model(data)

                rec_loss, kl_loss = self._vae_loss(data, z_phy_stat, z_aux2_stat, x)

                # Update metrics 
                self.valid_metrics.update('loss', rec_loss.item())
                # self.valid_metrics.update('rec_loss', rec_loss.item())
                self.valid_metrics.update('kl_loss', kl_loss.item())

        # Log the validation metrics
        self.logger.info(f"Validation Epoch: {epoch} Rec Loss: 
                         {rec_loss.item():.6f} KL Loss: {kl_loss.item():.6f}")

        return self.valid_metrics.result()

    def _vae_loss(self, data, z_phy_stat, z_aux2_stat, x):
        """
        Compute the VAE loss for the model.

        :param data: Input data.
        :param z_phy_stat: Dictionary with mean and log-variance for z_phy.
        :param z_aux2_stat: Dictionary with mean and log-variance for z_aux2.
        :param x: Reconstruction of the input data.
        :return: Reconstruction loss and KL divergence loss.
        """
        n = data.shape[0]
        rec_loss = torch.sum((x-data).pow(2)).mean()
        prior_z_phy_stat, prior_z_aux2_stat = self.model.priors(n, self.device) #TODO

        KL_z_phy = kldiv_normal_normal(z_phy_stat['mean'], z_phy_stat['lnvar'],
            prior_z_phy_stat['mean'], prior_z_phy_stat['lnvar']
            ) if not self.no_phy else torch.zeros(1, device=self.device)
        
        KL_z_aux2 = kldiv_normal_normal(z_aux2_stat['mean'], z_aux2_stat['lnvar'],
            prior_z_aux2_stat['mean'], prior_z_aux2_stat['lnvar']
            ) if self.config['arch']['phys_vae']['dim_z_aux2'] > 0 else torch.zeros(1, device=self.device)
        
        kl_loss = (KL_z_phy + KL_z_aux2).mean()

        return rec_loss, kl_loss
    
    def _unmixing_loss(self, unmixed, y):
        """
        Compute the unmixing regularization loss.

        :param unmixed: Physics-unmixed signal.
        :param y: Physics-only component generated by the physics model.
        :return: Unmixing loss.
        """
        if not self.no_phy:
            return torch.sum((unmixed - y.detach()).pow(2)).mean()
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
            return torch.mean((inferred_z_phy - synthetic_z_phy) ** 2)  # Regularization term
        else:
            return torch.zeros(1, device=self.device)

    def _least_action_loss(self, x_PB, x_P):
        """
        Compute the least action regularization loss.

        :param x_PB: Predicted physics-based signal.
        :param x_P: Predicted physics-only signal.
        :return: Least action loss.
        """
        if not self.no_phy:
            return torch.sum((x_PB - x_P).pow(2)).mean()
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