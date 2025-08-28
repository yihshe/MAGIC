"""
Simplified version of PhysVAE for ablation study and model inversion framework.

This implementation provides a cleaner, more interpretable version of the original PhysVAE
framework with the following key features:

1. **Simplified Architecture**: Removed unmixing path and complex components for clarity
2. **U-space Representation**: Physics parameters are represented in u-space (unbounded) 
   and transformed to z-space (0,1) via sigmoid
3. **Additive Residual Correction**: Uses gated additive residual instead of complex 
   multiplicative corrections
4. **Two-stage Training**: 
   - Stage A: Synthetic bootstrap for physics parameter learning
   - Stage B: Full VAE training with KL divergence
5. **Better Monitoring**: Enhanced logging and metrics for training stability

Key Changes from Original:
- KL divergence computed in u-space for physics parameters
- Simplified decoder with explicit additive residual and gate
- Removed unmixing path for cleaner ablation studies
- Better initialization of gate parameters
- Enhanced error handling and validation

TODO:
- Add more comprehensive ablation study configurations
- Implement additional physics models beyond RTM
- Add uncertainty quantification capabilities
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from base import BaseModel
from rtm_torch.rtm import RTM
from mogi.mogi import Mogi
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

        if not no_phy:
            if dim_z_aux >= 0:
                # phy+aux context -> expanded features in x-space
                self.func_context = MLP([dim_z_phy+dim_z_aux, 64, 128], activation)

                # CHANGED: residual head (corr) + gate
                self.func_residual = MLP([in_channels + 128, 128, in_channels], activation)
                self.func_gate = MLP([in_channels + 128, 64, in_channels], activation)
                # NEW: init last linear bias to ~ -4.6 so sigmoid ~ 0.01 at start
                self._init_gate_bias()
        else:
            # no phy
            self.func_aux_dec = MLP([dim_z_aux, 16, 32, 64, in_channels], activation)

    def _init_gate_bias(self):
        """Initialize gate bias to start with low gate values (~0.01)"""
        try:
            # Find the last linear layer in the gate network
            for module in reversed(list(self.func_gate.modules())):
                if isinstance(module, nn.Linear):
                    if module.bias is not None:
                        # Initialize bias to -4.6 so sigmoid(-4.6) ≈ 0.01
                        nn.init.constant_(module.bias, -4.6)
                    break
        except Exception as e:
            print(f"Warning: Could not initialize gate bias: {e}")
            pass

class FeatureExtractor(nn.Module):
    def __init__(self, config:dict):
        super(FeatureExtractor, self).__init__()

        in_channels = config['arch']['args']['input_dim']#11 for RTM, 36 for Mogi
        hidlayers_feat = config['arch']['phys_vae']['hidlayers_feat']#[32,]
        num_units_feat = config['arch']['phys_vae']['num_units_feat']#64
        activation = config['arch']['phys_vae']['activation']#elu
    
        # Shared backbone for feature extraction
        self.func_feat = MLP([in_channels,]+hidlayers_feat+[num_units_feat,], activation)

    def forward(self, x:torch.Tensor):
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

        # Decoding part
        self.dec = Decoders(config)

        # Encoding part
        self.enc = Encoders(config)

        # Physics
        self.physics_model = self.physics_init(config)
    
    def physics_init(self, config:dict):
        if config['arch']['args']['physics'] == 'RTM':
            return Physics_RTM(config)
        elif config['arch']['args']['physics'] == 'Mogi':
            return Physics_Mogi(config)
        else:
            raise ValueError("Unknown model type")
        
    def generate_physonly(self, z_phy:torch.Tensor, const:dict=None):
        # here z_phy is in (0,1)
        y = self.physics_model(z_phy, const=const) # (n, in_channels) 
        return y

    def priors(self, n:int, device:torch.device):
        """
        CHANGED: priors now in u-space for physics (standard normal),
        auxiliaries remain standard normal as before.
        """
        prior_u_phy_stat = {'mean': torch.zeros(n, self.dim_z_phy, device=device),
                            'lnvar': torch.zeros(n, self.dim_z_phy, device=device)}
        prior_z_aux_stat = {'mean': torch.zeros(n, max(0,self.dim_z_aux), device=device),
                            'lnvar': torch.zeros(n, max(0,self.dim_z_aux), device=device)}
        return prior_u_phy_stat, prior_z_aux_stat

    def encode(self, x:torch.Tensor):
        """
        CHANGED: no unmixing — infer u_phy directly from shared features.
        """
        x_ = x
        n = x_.shape[0]
        device = x_.device

        feature = self.enc.func_feat(x_)

        # infer z_aux
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

    def draw(self, z_phy_stat:dict, z_aux_stat:dict, hard_z:bool=False):
        """
        Sample in u-space, then squash to z in (0,1).
        """
        if not hard_z:
            u_phy = draw_normal(z_phy_stat['mean'], z_phy_stat['lnvar'])
            z_aux = draw_normal(z_aux_stat['mean'], z_aux_stat['lnvar'])
        else:
            u_phy = z_phy_stat['mean'].clone()
            z_aux = z_aux_stat['mean'].clone()

        if not self.no_phy:
            z_phy = torch.sigmoid(u_phy)  # CHANGED: no clamping
        else:
            z_phy = torch.zeros(u_phy.shape[0], self.in_channels, device=u_phy.device)

        return z_phy, z_aux

    def decode(self, z_phy:torch.Tensor, z_aux:torch.Tensor, full:bool=False, const:dict=None):
        """
        CHANGED: explicit additive residual with gate.
                 Inputs to corr & gate: concat([x_P, expanded]), where expanded depends on [z_phy, z_aux].
        """
        if not self.no_phy:
            y = self.physics_model(z_phy, const=const) # (n, in_channels)
            x_P = y
            if self.dim_z_aux >= 0:
                context = self.dec.func_context(torch.cat([z_phy.detach(), z_aux], dim=1))  # CHANGED: detach z_phy
                h = torch.cat([x_P, context], dim=1)
                delta = self.dec.func_residual(h)
                gate = torch.sigmoid(self.dec.func_gate(h))
                x_PB = x_P + gate * delta  # CHANGED: additive, gated
            else:
                x_PB = x_P.clone()
                gate = torch.zeros_like(x_P)
        else:
            y = torch.zeros(z_phy.shape[0], self.in_channels, device=z_phy.device)
            if self.dim_z_aux >= 0:
                x_PB = self.dec.func_aux_dec(z_aux) 
            else:
               x_PB = torch.zeros(z_phy.shape[0], self.in_channels, device=z_phy.device)
            x_P = x_PB.clone()
            gate = torch.zeros_like(x_PB)

        if full:
            return x_PB, x_P, y, gate  # CHANGED: return gate for logging/penalty
        else:
            return x_PB

    def forward(self, x:torch.Tensor, reconstruct:bool=True, hard_z:bool=False,
                inference:bool=False, const:dict=None):
        # CHANGED: encode returns only two dicts now
        z_phy_stat, z_aux_stat = self.encode(x)

        if not reconstruct:
            return z_phy_stat, z_aux_stat
        
        if not inference:
            x_mean = self.decode(*self.draw(z_phy_stat, z_aux_stat, hard_z=hard_z), full=False, const=const)
            return z_phy_stat, z_aux_stat, x_mean
        else:
            z_phy, z_aux = self.draw(z_phy_stat, z_aux_stat, hard_z=hard_z)
            x_PB, x_P, _y, _gate = self.decode(z_phy, z_aux, full=True, const=const)
            return z_phy, z_aux, x_PB, x_P  # keep 4-tuple for existing test code
