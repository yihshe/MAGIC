"""
PILA model (physics-informed low-rank augmentation).

This implementation provides the PILA architecture used in the paper.

To extend PILA with a new physics model, implement it under `physics/` and
create matching configs under `configs/phys_smpl/`.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from base import BaseModel
from physics.rtm.rtm import RTM
from physics.mogi.mogi import Mogi
from model import SCRIPT_DIR, PARENT_DIR
from utils import MLP, draw_normal

# CUDA for PyTorch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# RTM spectral bands to be used
S2_FULL_BANDS = ['B01', 'B02_BLUE', 'B03_GREEN', 'B04_RED','B05_RE1', 
                 'B06_RE2', 'B07_RE3', 'B08_NIR1', 'B8A_NIR2', 'B09_WV', 'B10', 
                 'B11_SWI1', 'B12_SWI2']
SD = 500.0 # Stem Density (SD), assumed to be 500 trees/ha

class Encoders(nn.Module):
    def __init__(self, config:dict):
        super(Encoders, self).__init__()

        in_channels = config['arch']['args']['input_dim'] #11 for RTM, 36 for Mogi
        no_phy = config['arch']['phys_vae']['no_phy']
        dim_z_aux = config['arch']['phys_vae']['dim_z_aux']#2
        dim_z_phy = config['arch']['phys_vae']['dim_z_phy']#7 for RTM, 4 for Mogi
        activation = config['arch']['phys_vae']['activation'] # e.g., 'elu'
        num_units_feat = config['arch']['phys_vae']['num_units_feat']#64
        
        self.func_feat = FeatureExtractor(config)

        if dim_z_aux > 0:
            hidlayers_z_aux = config['arch']['phys_vae']['hidlayers_z_aux']
            # z_aux encoding from feature only (as before)
            self.func_z_aux_mean = MLP([num_units_feat,]+hidlayers_z_aux+[dim_z_aux,], activation)
            self.func_z_aux_lnvar = MLP([num_units_feat,]+hidlayers_z_aux+[dim_z_aux,], activation)

        if not no_phy:
            hidlayers_z_phy = config['arch']['phys_vae']['hidlayers_z_phy']
            # CHANGED: remove Softplus on mean; final layer is linear (u-space)
            # ORIGINAL: self.func_z_phy_mean = nn.Sequential(MLP([...],[dim_z_phy]), nn.Softplus())
            self.func_z_phy_mean = MLP([num_units_feat,]+hidlayers_z_phy+[dim_z_phy,], activation)  # NEW (u-mean)
            self.func_z_phy_lnvar = MLP([num_units_feat,]+hidlayers_z_phy+[dim_z_phy,], activation) # u-lnvar

            # REMOVED: unmixing path; keep ablation-ready by simply not creating it now.
            # ORIGINAL: self.func_unmixer_coeff = nn.Sequential(MLP([...,[in_channels]]), nn.Tanh())

class Decoders(nn.Module):
    def __init__(self, config:dict):
        super(Decoders, self).__init__()

        in_channels = config['arch']['args']['input_dim'] #11 for RTM, 36 for Mogi
        dim_z_aux = config['arch']['phys_vae']['dim_z_aux'] #2
        dim_z_phy = config['arch']['phys_vae']['dim_z_phy'] #7 for RTM, 4 for Mogi
        activation = config['arch']['phys_vae']['activation'] #elu 
        no_phy = config['arch']['phys_vae']['no_phy']
        
        # Get time dimension and usage flags from config
        time_feat_dim = config['arch']['args'].get('time_feat_dim', 0)
        self.time_feat_dim = config['arch']['args'].get('time_feat_dim', 0)
        
        # Option to use time features in residual coefficient computation
        self.use_time_in_residual = config['arch']['args'].get('use_time_in_residual', False)

        if not no_phy:
            if dim_z_aux > 0:
                # IMPROVED: [z_aux, x_P_det] -> Linear -> c -> tanh(c/tau) -> delta = (c*s)@B.T
                residual_rank = config['arch']['phys_vae'].get('residual_rank', dim_z_aux)  # Default to dim_z_aux
                
                # Linear coefficient transformation: [z_aux, x_P_det, time_feats] -> residual_rank
                # Only add time_feat_dim if use_time_in_residual is True
                coeff_input_dim = dim_z_aux + in_channels + (time_feat_dim if self.use_time_in_residual else 0)
                self.coeff = nn.Linear(coeff_input_dim, residual_rank, bias=True)
                
                # Per-direction scale parameters
                self.s = nn.Parameter(torch.ones(residual_rank))  # per-direction scale
                
                # Basis matrix B: R^D x k (D = in_channels, k = residual_rank)
                self.B = nn.Parameter(torch.randn(in_channels, residual_rank))
                nn.init.orthogonal_(self.B)
                
                # Temperature annealing for coefficient computation
                self.tau_init = config['arch']['phys_vae'].get('tau_init', 3.0)  # Initial temperature
                self.tau_final = config['arch']['phys_vae'].get('tau_final', 1.0)  # Final temperature
                self.tau_warmup_epochs = config['arch']['phys_vae'].get('tau_warmup_epochs', 20)
                
                # Global residual scale warmup
                self.r_init = config['arch']['phys_vae'].get('r_init', 0.0)
                self.r_final = config['arch']['phys_vae'].get('r_final', 1.0)
                self.r_warmup_epochs = config['arch']['phys_vae'].get('r_warmup_epochs', 20)
                
                # Orthogonality penalty weight
                self.ortho_penalty_weight = config['arch']['phys_vae'].get('ortho_penalty_weight', 0.1)
            else:
                # When dim_z_aux = 0, set default values for tau/r methods to avoid errors
                self.tau_init = 1.0
                self.tau_final = 1.0
                self.tau_warmup_epochs = 0
                self.r_init = 0.0
                self.r_final = 0.0
                self.r_warmup_epochs = 0
        else:
            # no phy
            if dim_z_aux > 0:
                # Widths are config-driven so the no-physics baseline can be
                # sized to the input (11 S2 bands vs 1721 hyperspectral ones).
                hidlayers_aux_dec = config['arch']['phys_vae'].get('hidlayers_aux_dec', [16, 32, 64])
                self.func_aux_dec = MLP([dim_z_aux,]+hidlayers_aux_dec+[in_channels,], activation)
            else:
                # If dim_z_aux = 0, create a simple identity mapping
                self.func_aux_dec = nn.Identity()

    def get_tau(self, epoch, epochs_pretrain):
        """Get current temperature for annealing (starts after pretraining)"""
        if epoch < epochs_pretrain:
            return self.tau_init  # Keep initial temperature during pretraining
        elif epoch < epochs_pretrain + self.tau_warmup_epochs:
            progress = (epoch - epochs_pretrain) / self.tau_warmup_epochs
            return self.tau_init + progress * (self.tau_final - self.tau_init)
        return self.tau_final
    
    def get_r(self, epoch, epochs_pretrain):
        """Get current global residual scale (starts after pretraining)"""
        if epoch < epochs_pretrain:
            return self.r_init  # Keep initial scale during pretraining
        elif epoch < epochs_pretrain + self.r_warmup_epochs:
            progress = (epoch - epochs_pretrain) / self.r_warmup_epochs
            return self.r_init + progress * (self.r_final - self.r_init)
        return self.r_final
    
    def get_current_tau_r(self, epoch, epochs_pretrain):
        """Get current tau and r values for saving in checkpoint"""
        return {
            'tau': self.get_tau(epoch, epochs_pretrain),
            'r': self.get_r(epoch, epochs_pretrain),
            'epoch': epoch,
            'epochs_pretrain': epochs_pretrain
        }
    
    def set_tau_r_from_checkpoint(self, tau_r_dict):
        """Set tau and r values from checkpoint for inference"""
        if tau_r_dict is not None:
            self._inference_tau = tau_r_dict.get('tau', self.tau_final)
            self._inference_r = tau_r_dict.get('r', self.r_final)
        else:
            self._inference_tau = self.tau_final
            self._inference_r = self.r_final
    
    def get_tau_for_inference(self):
        """Get tau value for inference (uses saved value if available)"""
        return getattr(self, '_inference_tau', self.tau_final)
    
    def get_r_for_inference(self):
        """Get r value for inference (uses saved value if available)"""
        return getattr(self, '_inference_r', self.r_final)
    
    def compute_coefficient(self, z_aux, x_P_det, epoch, epochs_pretrain, use_inference_values=False, time_feats=None):
        """
        IMPROVED: Compute coefficient from [z_aux, x_P_det, time_feats] with temperature annealing
        c_raw = Linear([z_aux, x_P_det, time_feats])
        c = tanh(c_raw / tau)
        """
        # Concatenate z_aux, physics context, and time features
        if time_feats is not None and self.time_feat_dim > 0 and self.use_time_in_residual:
            # Use only the first time_feat_dim elements of the 4-dim time features
            time_feats_sliced = time_feats[..., :self.time_feat_dim]
            coeff_input = torch.cat([z_aux, x_P_det, time_feats_sliced], dim=1)
        else:
            coeff_input = torch.cat([z_aux, x_P_det], dim=1)
        c_raw = self.coeff(coeff_input)  # (batch_size, residual_rank)
        
        # Apply temperature annealing (starts after pretraining)
        if use_inference_values:
            tau = self.get_tau_for_inference()
        else:
            tau = self.get_tau(epoch, epochs_pretrain)
        c = torch.tanh(c_raw / tau)  # (batch_size, residual_rank)
        return c
    
    def orthogonality_penalty(self):
        """Compute orthogonality penalty for basis matrix B"""
        BtB = torch.matmul(self.B.T, self.B)
        I = torch.eye(self.B.shape[1], device=self.B.device)
        return torch.norm(BtB - I, p='fro') ** 2

class FeatureExtractor(nn.Module):
    def __init__(self, config:dict):
        super(FeatureExtractor, self).__init__()

        in_channels = config['arch']['args']['input_dim']#11 for RTM, 36 for Mogi
        hidlayers_feat = config['arch']['phys_vae']['hidlayers_feat']#[32,]
        num_units_feat = config['arch']['phys_vae']['num_units_feat']#64
        activation = config['arch']['phys_vae']['activation']#elu
        
        # Feature extractor projects in_channels -> num_units_feat
        self.func_feat = MLP([in_channels,]+hidlayers_feat+[num_units_feat,], activation)

    def forward(self, x:torch.Tensor, t:torch.Tensor=None):
        # Time features are not used in the feature extractor input
        return self.func_feat(x) # n x num_units_feat


class Physics_RTM(nn.Module):
    def __init__(self, config:dict):
        super(Physics_RTM, self).__init__()
        self.model = RTM()
        self.z_phy_ranges = json.load(open(os.path.join(PARENT_DIR, config['arch']['args']['rtm_paras']), 'r'))
        self.bands_index = [i for i in range(
            len(S2_FULL_BANDS)) if S2_FULL_BANDS[i] not in ['B01', 'B10']]
        # Mean and scale for standardization
        self.x_mean = torch.tensor(
            np.load(os.path.join(PARENT_DIR,config['arch']['args']['standardization']['x_mean']))
            ).float().unsqueeze(0).to(DEVICE)
        self.x_scale = torch.tensor(
            np.load(os.path.join(PARENT_DIR, config['arch']['args']['standardization']['x_scale']))
            ).float().unsqueeze(0).to(DEVICE)
    
    def rescale(self, z_phy:torch.Tensor):
        """
        Rescale z in (0,1) to physical parameters in original scales.
        """
        z_phy_rescaled = {}
        for i, para_name in enumerate(self.z_phy_ranges.keys()):
            z_phy_rescaled[para_name] = z_phy[:, i] * (
                self.z_phy_ranges[para_name]['max'] - self.z_phy_ranges[para_name]['min']
                ) + self.z_phy_ranges[para_name]['min']
        
        z_phy_rescaled['cd'] = torch.sqrt(
            (z_phy_rescaled['fc']*10000)/(torch.pi*SD))*2
        z_phy_rescaled['h'] = torch.exp(
            2.117 + 0.507*torch.log(z_phy_rescaled['cd'])) 
        
        return z_phy_rescaled
    
    def forward(self, z_phy:torch.Tensor, const:dict=None):
        z_phy_rescaled = self.rescale(z_phy)
        if const is not None:
            z_phy_rescaled.update(const)
        output = self.model.run(**z_phy_rescaled)[:, self.bands_index]
        return (output - self.x_mean) / self.x_scale 

class Physics_PROSAIL(nn.Module):
    """
    PROSPECT-D + 4SAIL over the 1721 GreenHyperSpectra bands.

    Differs from Physics_RTM only in the RTM configuration (leaf model, canopy
    model, no sensor resampling) and in that some leaf parameters are held
    constant rather than inferred.

    PROSPECT-D rather than PROSPECT-PRO: PRO resolves dry matter into protein
    and carbon-based constituents, but that extra degree of freedom is not
    identifiable from fresh canopy spectra. Its coefficients are near-collinear
    (cos(K_cbc, K_cm) = 0.9986) and protein is ~9x scarcer than cbc, leaving a
    ~5% perturbation; the GHS labels satisfy cm == cp + cbc to 1e-7, so only two
    degrees of freedom are present in the first place. PROSPECT-D takes cm
    directly.

    `rtm_constants` (optional) pins RTM parameters that are present in the leaf
    model but not retrievable -- here carotenoid and anthocyanin, whose
    absorption is largely masked by chlorophyll. They are passed to RTM.run
    alongside the learnable ones, so nothing in physics/rtm needs to change and
    RTM.para_dict keeps its defaults for everything left unspecified (N = 1.5).
    """
    def __init__(self, config:dict):
        super(Physics_PROSAIL, self).__init__()
        self.model = RTM()
        # 'default' sensor -> no SRF resampling, full 1 nm 400-2500 nm output
        self.model.model_init(sensor='default', lop='prospectD', canopy_arch='sail')
        self.z_phy_ranges = json.load(open(os.path.join(PARENT_DIR, config['arch']['args']['rtm_paras']), 'r'))
        const_path = config['arch']['args'].get('rtm_constants', None)
        self.rtm_constants = json.load(open(os.path.join(PARENT_DIR, const_path), 'r')) \
            if const_path else {}
        # GHS keeps 400-1350, 1431-1800 and 2051-2450 nm; the RTM emits 400-2500
        # nm at 1 nm, so band i corresponds to wavelength 400 + i.
        self.bands_index = list(range(0, 951)) + list(range(1031, 1401)) + list(range(1651, 2051))
        # Mean and scale for standardization
        self.x_mean = torch.tensor(
            np.load(os.path.join(PARENT_DIR, config['arch']['args']['standardization']['x_mean']))
            ).float().unsqueeze(0).to(DEVICE)
        self.x_scale = torch.tensor(
            np.load(os.path.join(PARENT_DIR, config['arch']['args']['standardization']['x_scale']))
            ).float().unsqueeze(0).to(DEVICE)

    def rescale(self, z_phy:torch.Tensor):
        """Rescale z in (0,1) to the learnable physical parameters."""
        z_phy_rescaled = {}
        for i, para_name in enumerate(self.z_phy_ranges.keys()):
            z_phy_rescaled[para_name] = z_phy[:, i] * (
                self.z_phy_ranges[para_name]['max'] - self.z_phy_ranges[para_name]['min']
                ) + self.z_phy_ranges[para_name]['min']

        return z_phy_rescaled

    def forward(self, z_phy:torch.Tensor, const:dict=None):
        z_phy_rescaled = self.rescale(z_phy)
        # Fixed leaf parameters, expanded to the batch so RTM.para_reset treats
        # them like the learnable ones (it does para_dict.update(paras), so no
        # change to physics/rtm is needed). Per-sample `const` wins if given.
        for name, value in self.rtm_constants.items():
            z_phy_rescaled[name] = torch.full(
                (z_phy.shape[0],), float(value),
                dtype=z_phy.dtype, device=z_phy.device)
        if const is not None:
            z_phy_rescaled.update(const)
        # RTM.para_reset broadcasts the frozen parameters to the batch size of
        # the first call only, which then caps every later batch. Re-arming the
        # flag makes this physics module batch-size agnostic (it costs one
        # torch.full per frozen parameter) so that training, validation and
        # evaluation can use different batch sizes in any order.
        self.model.reset_non_learnable = True
        output = self.model.run(**z_phy_rescaled)[:, self.bands_index]
        return (output - self.x_mean) / self.x_scale

class Physics_Mogi(nn.Module):
    def __init__(self, config:dict):
        super(Physics_Mogi, self).__init__()

        self.z_phy_ranges = json.load(open(os.path.join(PARENT_DIR, config['arch']['args']['mogi_paras']), 'r'))
        self.station_info = json.load(open(os.path.join(PARENT_DIR, config['arch']['args']['station_info']), 'r'))
        
        x = torch.tensor([self.station_info[k]['xE']
                         for k in self.station_info.keys()])*1000  # m
        y = torch.tensor([self.station_info[k]['yN']
                         for k in self.station_info.keys()])*1000  # m
        self.model = Mogi(x,y)
        
        # Mean and scale for standardization
        self.x_mean = torch.tensor(
            np.load(os.path.join(PARENT_DIR,config['arch']['args']['standardization']['x_mean']))
            ).float().unsqueeze(0).to(DEVICE)
        self.x_scale = torch.tensor(
            np.load(os.path.join(PARENT_DIR, config['arch']['args']['standardization']['x_scale']))
            ).float().unsqueeze(0).to(DEVICE)
    
    def rescale(self, z_phy:torch.Tensor):
        """
        Rescale z in (0,1) to physical parameters in the original scale.
        """
        z_phy_rescaled = {}
        for i, para_name in enumerate(self.z_phy_ranges.keys()):
            minv = self.z_phy_ranges[para_name]['min']
            maxv = self.z_phy_ranges[para_name]['max']
            if len(z_phy.shape) == 3:
                z_phy_rescaled[para_name] = z_phy[:, :, i]*(maxv-minv)+minv
            else:
                z_phy_rescaled[para_name] = z_phy[:, i]*(maxv-minv)+minv

            if para_name in ['xcen', 'ycen', 'd']:
                z_phy_rescaled[para_name] = z_phy_rescaled[para_name]*1000

        z_phy_rescaled['dV'] = z_phy_rescaled['dV'] * \
            torch.pow(10, torch.tensor(5)) - torch.pow(10, torch.tensor(7))
        
        return z_phy_rescaled
    
    def forward(self, z_phy:torch.Tensor, const:dict=None):
        z_phy_rescaled = self.rescale(z_phy)
        output = self.model.run(**z_phy_rescaled)
        return (output - self.x_mean) / self.x_scale 

class PHYS_VAE_SMPL(nn.Module):
    def __init__(self, config:dict):
        super(PHYS_VAE_SMPL, self).__init__()

        self.no_phy = config['arch']['phys_vae']['no_phy']
        self.dim_z_aux = config['arch']['phys_vae']['dim_z_aux']
        self.dim_z_phy = config['arch']['phys_vae']['dim_z_phy']
        self.activation = config['arch']['phys_vae']['activation']
        self.in_channels = config['arch']['args']['input_dim']
        self.detach_x_P_for_bias = config['arch']['phys_vae'].get('detach_x_P_for_bias', True)

        # EMA prior configuration
        self.use_ema_prior = config['trainer']['phys_vae'].get('use_ema_prior', False)
        self.ema_momentum = config['trainer']['phys_vae'].get('ema_momentum', 0.99)
        
        # EMA variance bounds (hardcoded)
        self.ema_min_var = 1e-3  # variance floor
        self.ema_max_var = 50.0  # variance ceiling

        # Encoding part
        self.enc = Encoders(config)

        # Decoding part
        self.dec = Decoders(config)

        # Physics
        self.physics_model = self.physics_init(config)
        
        # EMA buffers for u_phy statistics (only if EMA prior is enabled)
        if self.use_ema_prior and not self.no_phy:
            self.register_buffer('ema_mean', torch.zeros(self.dim_z_phy))  # E[U]
            self.register_buffer('ema_m2', torch.ones(self.dim_z_phy))     # E[U^2]
            self.register_buffer('ema_var', torch.ones(self.dim_z_phy))    # convenience buffer
        
        # Store time features for decoder use
        self.time_feats = None
    
    def physics_init(self, config:dict):
        if config['arch']['args']['physics'] == 'RTM':
            return Physics_RTM(config)
        elif config['arch']['args']['physics'] == 'PROSAIL':
            return Physics_PROSAIL(config)
        elif config['arch']['args']['physics'] == 'Mogi':
            return Physics_Mogi(config)
        else:
            raise ValueError("Unknown model type")
        
    def generate_physonly(self, z_phy:torch.Tensor, const:dict=None):
        # here z_phy is in (0,1)
        y = self.physics_model(z_phy, const=const) # (n, in_channels) 
        return y

    def update_ema_prior(self, u_phy_mean: torch.Tensor, u_phy_lnvar: torch.Tensor):
        """
        Update EMA statistics for u_phy using exponential moving average.
        
        Args:
            u_phy_mean: Current batch posterior means (shape: [batch_size, dim_z_phy])
            u_phy_lnvar: Current batch posterior log-variances (shape: [batch_size, dim_z_phy])
        """
        if not self.use_ema_prior or self.no_phy:
            return
            
        with torch.no_grad():
            decay = self.ema_momentum  # e.g., 0.999
            mu_q = u_phy_mean.detach()                 # (B, D)
            var_q = torch.exp(u_phy_lnvar.detach())    # (B, D)

            # First moment E[U]
            batch_mu = mu_q.mean(dim=0)                # (D,)

            # Second moment E[U^2] = E[var + mu^2]
            batch_m2 = (var_q + mu_q**2).mean(dim=0)   # (D,)

            # EMA updates
            self.ema_mean.mul_(decay).add_(batch_mu, alpha=1 - decay)
            self.ema_m2.mul_(decay).add_(batch_m2, alpha=1 - decay)

            # Convert moments to variance and clamp
            ema_var = (self.ema_m2 - self.ema_mean**2).clamp(self.ema_min_var, self.ema_max_var)
            self.ema_var.copy_(ema_var)

    def priors(self, n:int, device:torch.device):
        """
        CHANGED: priors now in u-space for physics (standard normal or EMA Gaussian),
        auxiliaries remain standard normal as before.
        """
        if self.use_ema_prior and not self.no_phy:
            # Use EMA Gaussian prior for u_phy
            ema_var = (self.ema_m2 - self.ema_mean**2).clamp(self.ema_min_var, self.ema_max_var)
            prior_u_phy_stat = {
                'mean': self.ema_mean.unsqueeze(0).expand(n, -1),
                'lnvar': ema_var.log().unsqueeze(0).expand(n, -1)
            }
        else:
            # Use standard normal prior for u_phy
            prior_u_phy_stat = {'mean': torch.zeros(n, self.dim_z_phy, device=device),
                                'lnvar': torch.zeros(n, self.dim_z_phy, device=device)}
        
        prior_z_aux_stat = {'mean': torch.zeros(n, max(0,self.dim_z_aux), device=device),
                            'lnvar': torch.zeros(n, max(0,self.dim_z_aux), device=device)}
        return prior_u_phy_stat, prior_z_aux_stat

    def encode(self, x:torch.Tensor, t:torch.Tensor=None):
        """
        CHANGED: z_aux encoding from feature only (as before).
        Now supports optional time features.
        """
        x_ = x
        n = x_.shape[0]
        device = x_.device

        # Store time features for decoder use
        self.time_feats = t

        feature = self.enc.func_feat(x_, t)

        # infer z_aux from feature only
        if self.dim_z_aux > 0:
            z_aux_stat = {'mean': self.enc.func_z_aux_mean(feature),
                          'lnvar': self.enc.func_z_aux_lnvar(feature)}
        else:
            z_aux_stat = {'mean': torch.empty(n, 0, device=device),
                          'lnvar': torch.empty(n, 0, device=device)}

        # infer u_phy stats (stored in z_phy_stat for backward-compat)
        if not self.no_phy:
            z_phy_stat = {'mean': self.enc.func_z_phy_mean(feature),   # u-mean
                          'lnvar': self.enc.func_z_phy_lnvar(feature)} # u-lnvar
        else:
            z_phy_stat = {'mean': torch.empty(n, 0, device=device),
                          'lnvar': torch.empty(n, 0, device=device)}

        return z_phy_stat, z_aux_stat

    def draw(self, z_phy_stat:dict, z_aux_stat:dict, hard_z_phy:bool=False, hard_z_aux:bool=False):
        """
        Sample in u-space, then squash to z in (0,1).
        z_aux remains in u-space (unbounded) for coefficient computation.
        
        Args:
            hard_z_phy: If True, use deterministic sampling for z_phy (mean)
            hard_z_aux: If True, use deterministic sampling for z_aux (mean)
        """
        # Sample z_phy based on its individual setting
        if not hard_z_phy:
            u_phy = draw_normal(z_phy_stat['mean'], z_phy_stat['lnvar'])
        else:
            u_phy = z_phy_stat['mean'].clone()
            
        # Sample z_aux based on its individual setting
        if not hard_z_aux:
            z_aux = draw_normal(z_aux_stat['mean'], z_aux_stat['lnvar'])  # Keep in u-space
        else:
            z_aux = z_aux_stat['mean'].clone()

        if not self.no_phy:
            z_phy = torch.sigmoid(u_phy)  # CHANGED: no clamping
        else:
            z_phy = torch.zeros(u_phy.shape[0], self.in_channels, device=u_phy.device)

        return z_phy, z_aux  # Return z_aux (unbounded) instead of z_aux

    def decode(self, z_phy:torch.Tensor, z_aux:torch.Tensor, epoch:int=0, epochs_pretrain:int=20, full:bool=False, const:dict=None, use_inference_values:bool=False, detach_x_P_for_bias:bool=True):
        """
        CORRECTED: z_aux (unbounded) -> c = tanh(z_aux/tau) -> delta = c@B.T
        x_PB = x_P + r(t) * delta
        """
        if not self.no_phy:
            y = self.physics_model(z_phy, const=const) # (n, in_channels)
            x_P = y
            if self.dim_z_aux > 0:
                # Compute coefficient from z_aux with temperature annealing
                x_P_input = x_P.detach() if detach_x_P_for_bias else x_P
                # Pass time features if available
                c = self.dec.compute_coefficient(z_aux, x_P_input, epoch, epochs_pretrain, use_inference_values, self.time_feats)
                
                # Compute low-rank residual: delta = (c * s) @ B.T
                delta = torch.matmul(c * self.dec.s, self.dec.B.T)
                
                # Apply global residual scale warmup (starts after pretraining)
                if use_inference_values:
                    r = self.dec.get_r_for_inference()
                else:
                    r = self.dec.get_r(epoch, epochs_pretrain)
                x_PB = x_P + r * delta
            else:
                x_PB = x_P.clone()
                delta = torch.zeros_like(x_P)
                c = torch.zeros(x_P.shape[0], 0, device=x_P.device)
        else:
            y = torch.zeros(z_phy.shape[0], self.in_channels, device=z_phy.device)
            if self.dim_z_aux > 0:
                x_PB = self.dec.func_aux_dec(z_aux) 
            else:
               x_PB = torch.zeros(z_phy.shape[0], self.in_channels, device=z_phy.device)
            x_P = x_PB.clone()
            delta = torch.zeros_like(x_PB)
            c = torch.zeros(x_P.shape[0], 0, device=x_P.device)

        if full:
            return x_PB, x_P, y, delta, c
        else:
            return x_PB

    def forward(self, x:torch.Tensor, t:torch.Tensor=None, reconstruct:bool=True, hard_z_phy:bool=False, hard_z_aux:bool=False,
                inference:bool=False, const:dict=None, epoch:int=0, epochs_pretrain:int=20):
        z_phy_stat, z_aux_stat = self.encode(x, t)

        if not reconstruct:
            return z_phy_stat, z_aux_stat
        
        if not inference:
            x_mean = self.decode(*self.draw(z_phy_stat, z_aux_stat, hard_z_phy=hard_z_phy, hard_z_aux=hard_z_aux), 
                               epoch=epoch, epochs_pretrain=epochs_pretrain, full=False, const=const, use_inference_values=False, detach_x_P_for_bias=self.detach_x_P_for_bias)
            return z_phy_stat, z_aux_stat, x_mean
        else:
            z_phy, z_aux = self.draw(z_phy_stat, z_aux_stat, hard_z_phy=hard_z_phy, hard_z_aux=hard_z_aux)
            x_PB, x_P, _y, _delta, _c = self.decode(z_phy, z_aux, epoch=epoch, epochs_pretrain=epochs_pretrain, full=True, const=const, use_inference_values=True, detach_x_P_for_bias=self.detach_x_P_for_bias)
            return z_phy, z_aux, x_PB, x_P  # keep 4-tuple for existing test code
