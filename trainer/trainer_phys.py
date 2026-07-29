7# Adapted from the training script of Phys-VAE
import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, kldiv_normal_normal
import wandb
from model.loss import mse_loss, mse_loss_per_channel, end_stop_penalty
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
        # self.kl_loss_weight = config['trainer']['phys_vae']['balance_kld']
        self.unmix_loss_weight = config['trainer']['phys_vae']['balance_unmix']
        self.synthetic_data_loss_weight = config['trainer']['phys_vae']['balance_data_aug']
        self.least_action_loss_weight = config['trainer']['phys_vae']['balance_lact_dec']

        # === [REBUTTAL ONLY] not part of the published HVAE runs ==============
        # Reviewer Q: "What about using the PILA end-stop loss in the HVAE method
        # -- do PILA's benefits come from the architecture or the prior?"
        # Both knobs below default to the published behaviour and no config under
        # configs/ sets either key, so those runs are bit-identical. Used only by
        # rebuttal/prior_vs_endstop/configs/.
        # ----------------------------------------------------------------------
        # Per-latent KL weights. Absent -> the single combined weight above is
        # applied to (KL_z_phy + KL_z_aux2) exactly as before. Present -> the two
        # KL terms are weighted separately, which is what lets z_phy's prior be
        # switched off (balance_kld_z_phy = 0) without touching z_aux2's.
        self.kl_weight_z_phy = config['trainer']['phys_vae'].get('balance_kld_z_phy', None)
        self.kl_weight_z_aux = config['trainer']['phys_vae'].get('balance_kld_z_aux', None)
        self.split_kl = (self.kl_weight_z_phy is not None) or (self.kl_weight_z_aux is not None)
        if self.split_kl:
            if self.kl_weight_z_phy is None:
                self.kl_weight_z_phy = self.kl_loss_weight
            if self.kl_weight_z_aux is None:
                self.kl_weight_z_aux = self.kl_loss_weight

        # PILA's end-stop penalty on z_phy. 0.0 -> the term is never computed.
        # target: 'raw' penalises the pre-clamp draw (default: the clamped value
        # carries no gradient at the end-stops, see model.loss.end_stop_penalty),
        # 'clamped' penalises the value the decoder actually receives.
        self.edge_penalty_weight = config['trainer']['phys_vae'].get('edge_penalty_weight', 0.0)
        self.edge_penalty_power = config['trainer']['phys_vae'].get('edge_penalty_power', 1.0)
        # Dynamic weighting, as HVAE already does for its unmix / synthetic-data /
        # least-action terms: _loss_weight() rescales the term to sit one order of
        # magnitude below rec_loss, so it stays subordinate as rec_loss falls and
        # no absolute weight has to be guessed. edge_penalty_weight then only acts
        # as an on/off switch (any value > 0 enables the term).
        self.edge_penalty_dynamic = config['trainer']['phys_vae'].get('edge_penalty_dynamic', False)
        self.edge_penalty_target = config['trainer']['phys_vae'].get('edge_penalty_target', 'raw')
        self.edge_penalty_eps = config['trainer']['phys_vae'].get('edge_penalty_eps', 1e-2)
        # ======================================================================

        # Metric tracker
        self.train_metrics = MetricTracker(
            'loss', 'rec_loss', 'kl_loss', 'unmix_loss', 'syn_data_loss', 'least_act_loss', 'smoothness_loss',
            'kl_z_phy', 'kl_z_aux',      # the two KL terms, tracked separately
            'edge_penalty',              # end-stop penalty (0 unless enabled)
            'edge_penalty_weight',       # its applied weight (varies if dynamic)
            'z_phy_clamp_frac',          # fraction of z_phy entries the clamp had to cut
            *[m.__name__ for m in metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker(
            'loss', 'kl_loss',
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

        # Reconstruction loss is summed over the feature axis, so its magnitude
        # scales with the number of channels (11 S2 bands vs 1721 hyperspectral
        # ones). Scaling it here keeps every other loss weight transferable
        # instead of re-tuning them per dataset. Default 1.0 = unchanged.
        self.rec_loss_scale = config['trainer'].get('rec_loss_scale', 1.0)
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
        sequence_len = None

        for batch_idx, data_dict in enumerate(self.data_loader):
            data = data_dict[self.data_key].to(self.device)
            # TODO load the angle specific information here
            if self.input_const_keys is not None:
                input_const = {k: data_dict[k].to(self.device) for k in self.input_const_keys}
            else:
                input_const = None

            if data.dim() == 3:
                sequence_len = data.size(1)
                data = data.view(-1, data.size(-1))

            self.optimizer.zero_grad()

            # Encode step: Infer latent variables
            z_phy_stat, z_aux2_stat, unmixed = self.model.encode(data)

            # Draw step: Sample latent variables
            if self.edge_penalty_weight > 0:
                z_phy, z_aux2, z_phy_raw = self.model.draw(z_phy_stat, z_aux2_stat, hard_z=False, return_raw=True)
            else:
                z_phy, z_aux2 = self.model.draw(z_phy_stat, z_aux2_stat, hard_z=False)
                z_phy_raw = z_phy

            # Decode step: Reconstruct outputs
            x_PB, x_P, x_lnvar, y = self.model.decode(z_phy, z_aux2, full=True, const = input_const)

            # Reconstruction variance
            x_var = torch.exp(x_lnvar)

            # Loss calculations
            # ELBO loss #TODO element-wise loss, not sample-wise
            rec_loss, kl_loss, kl_parts = self._vae_loss(data, z_phy_stat, z_aux2_stat, x_PB)

            # Weighted KL. Published path (no per-latent weights configured) is
            # the original single-weight expression, unchanged; the split branch
            # is [REBUTTAL ONLY].
            if self.split_kl:
                kl_weighted = self.kl_weight_z_phy * kl_parts['z_phy'] + self.kl_weight_z_aux * kl_parts['z_aux']
            else:
                kl_weighted = self.kl_loss_weight * kl_loss

            # [REBUTTAL ONLY] end-stop penalty on z_phy; returns 0 unless enabled
            edge_penalty = self._edge_penalty(z_phy_raw if self.edge_penalty_target == 'raw' else z_phy)
            edge_w = 0.0  # set below when the term is active

            # Unmixing regularization (R_{DA,1})
            unmix_loss = self._unmixing_loss(unmixed, y)  # Unmixing regularization TODO reg_unmix (R_{DA,1})

            # Synthetic data regularization (R_{DA,2})
            synthetic_data_loss = self._synthetic_data_loss(data.shape[0])

            # Least action regularization (R_{ppc}) 
            least_action_loss = self._least_action_loss(x_PB, x_P) 

            # Sequence smoothness regularization (for Mogi model)
            smoothness_loss = self._sequence_smoothness_loss(x_P, sequence_len)

            
            # Total loss 
            if not self.no_phy and epoch < self.epochs_pretrain: 
                # Pretraining phase, not used in default setting
                loss = self.synthetic_data_loss_weight * synthetic_data_loss
            else:
                # Training phase TODO loss weights TBD
                # loss = rec_loss \
                #     + self.kl_loss_weight * kl_loss * x_var.detach() \
                #     + self.unmix_loss_weight * unmix_loss \
                #     + self.synthetic_data_loss_weight * synthetic_data_loss \
                #     + self.least_action_loss_weight * least_action_loss
                
                # NOTE for experiments before 2025.04.06, + self.kl_loss_weight * kl_loss * x_var.detach() \
                loss = rec_loss \
                    + kl_weighted * x_var.detach() \
                    + self._loss_weight(rec_loss, unmix_loss) * unmix_loss \
                    + self._loss_weight(rec_loss, synthetic_data_loss) * synthetic_data_loss \
                    + self._loss_weight(rec_loss, least_action_loss) * least_action_loss \
                    + 0.01 * smoothness_loss

                # [REBUTTAL ONLY] appended rather than folded into the expression
                # above so configs without an end-stop weight (i.e. every
                # published one) keep bit-identical losses.
                if self.edge_penalty_weight > 0:
                    edge_w = float(self._loss_weight(rec_loss, edge_penalty)) if self.edge_penalty_dynamic \
                        else self.edge_penalty_weight
                    loss = loss + edge_w * edge_penalty


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
            self.train_metrics.update('smoothness_loss', smoothness_loss.item())
            self.train_metrics.update('kl_z_phy', kl_parts['z_phy'].item())
            self.train_metrics.update('kl_z_aux', kl_parts['z_aux'].item())
            self.train_metrics.update('edge_penalty', edge_penalty.item())
            self.train_metrics.update('edge_penalty_weight', edge_w)
            # How often the clamp is what keeps z_phy in range -- the diagnostic
            # that says whether the end-stop penalty (or the prior) is doing the work.
            with torch.no_grad():
                clamped = ((z_phy_raw < 0.0) | (z_phy_raw > 1.0)).float().mean() if not self.no_phy else torch.zeros(1, device=self.device)
            self.train_metrics.update('z_phy_clamp_frac', clamped.item())

            # Logging
            if batch_idx % self.config['trainer']['log_step'] == 0:
                log_str = (f"Train Epoch: {epoch} [{batch_idx}/{len(self.data_loader)}] "
                           f"Loss: {loss.item():.6f} Rec Loss: {rec_loss.item():.6f} "
                           f"KL Loss: {kl_loss.item():.6f} Unmix Loss: {unmix_loss.item():.6f} "
                           f"Synthetic Data Loss: {synthetic_data_loss.item():.6f} "
                           f"Least Action Loss: {least_action_loss.item():.6f} "
                           f"Smoothness Loss: {smoothness_loss.item():.6f}")
                if self.split_kl or self.edge_penalty_weight > 0:
                    log_str += (f" KL_z_phy: {kl_parts['z_phy'].item():.6f} "
                                f"KL_z_aux: {kl_parts['z_aux'].item():.6f} "
                                f"Edge_pen: {edge_penalty.item():.6f} "
                                f"(w {edge_w:.4g}) "
                                f"clamp_frac: {clamped.item():.4f}")
                self.logger.info(log_str)

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

                z_phy_stat, z_aux2_stat, x, _ = self.model(data, const = input_const)

                rec_loss, kl_loss, _ = self._vae_loss(data, z_phy_stat, z_aux2_stat, x)

                # Update metrics 
                self.valid_metrics.update('loss', rec_loss.item())
                # self.valid_metrics.update('rec_loss', rec_loss.item())
                self.valid_metrics.update('kl_loss', kl_loss.item())

        # Log the validation metrics
        self.logger.info(f"Validation Epoch: {epoch} Rec Loss: {rec_loss.item():.6f} KL Loss: {kl_loss.item():.6f}")

        return self.valid_metrics.result()

    def _vae_loss(self, data, z_phy_stat, z_aux2_stat, x):
        """
        Compute the VAE loss for the model.

        :param data: Input data.
        :param z_phy_stat: Dictionary with mean and log-variance for z_phy.
        :param z_aux2_stat: Dictionary with mean and log-variance for z_aux2.
        :param x: Reconstruction of the input data.
        :return: Reconstruction loss, combined KL divergence loss, and the two KL
                 terms separately (so they can be weighted independently).
        """
        n = data.shape[0]
        rec_loss = torch.sum((x-data).pow(2), dim=1).mean() * self.rec_loss_scale
        prior_z_phy_stat, prior_z_aux2_stat = self.model.priors(n, self.device) #TODO

        KL_z_phy = kldiv_normal_normal(z_phy_stat['mean'], z_phy_stat['lnvar'],
            prior_z_phy_stat['mean'], prior_z_phy_stat['lnvar']
            ) if not self.no_phy else torch.zeros(1, device=self.device)
        
        KL_z_aux2 = kldiv_normal_normal(z_aux2_stat['mean'], z_aux2_stat['lnvar'],
            prior_z_aux2_stat['mean'], prior_z_aux2_stat['lnvar']
            ) if self.config['arch']['phys_vae']['dim_z_aux2'] > 0 else torch.zeros(1, device=self.device)
        
        kl_loss = (KL_z_phy + KL_z_aux2).mean()
        kl_parts = {'z_phy': KL_z_phy.mean(), 'z_aux': KL_z_aux2.mean()}

        return rec_loss, kl_loss, kl_parts

    def _edge_penalty(self, z_phy):
        """
        [REBUTTAL ONLY] End-stop penalty on z_phy, transplanted from PILA (R_end).
        Not part of the published HVAE baseline; inactive unless
        trainer.phys_vae.edge_penalty_weight > 0.

        Identical barrier to PhysVAETrainerSMPL._edge_penalty; the difference is
        that HVAE's z_phy is not bounded by construction, so the barrier is
        continued linearly past the end-stops (see model.loss.end_stop_penalty).

        :param z_phy: Physical latents, either the pre-clamp draw or the clamped one.
        :return: Scalar penalty (zero when disabled or when there is no physics).
        """
        if self.no_phy or self.edge_penalty_weight <= 0 or z_phy is None or z_phy.numel() == 0:
            return torch.zeros(1, device=self.device).squeeze()
        return end_stop_penalty(z_phy, power=self.edge_penalty_power, eps=self.edge_penalty_eps)


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
        if not self.no_phy:
            return torch.sum((x_PB - x_P).pow(2), dim=1).mean()
        else:
            return torch.zeros(1, device=self.device)
    
    def _sequence_smoothness_loss(self, x_P, sequence_len=None):
        """
        Compute the smoothness regularization loss for a sequence.

        :param x_P: Predicted physics-only signal.
        :param sequence_len: Length of the sequence.
        :return: Smoothness loss.
        """
        if not self.no_phy and sequence_len is not None:
            x_P_reshaped = x_P.view(-1, sequence_len, x_P.size(-1))
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