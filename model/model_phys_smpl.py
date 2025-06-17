"""
Simplified version of PhysVAE for ablation study.
TODO - Rename some variables to avoid confusion
TODO - Remove unnecessary lines of code
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
        activation = config['arch']['phys_vae']['activation'] #same as galaxy example for gaussian sampling
        num_units_feat = config['arch']['phys_vae']['num_units_feat']#64
        
        self.func_feat = FeatureExtractor(config)

        if dim_z_aux > 0:
            # hidlayers_aux_enc = config['arch']['phys_vae']['hidlayers_aux_enc'] #[64, 32] TODO change the config too
            hidlayers_z_aux = config['arch']['phys_vae']['hidlayers_z_aux'] #[64, 32]
            self.func_z_aux_mean = MLP([num_units_feat,]+hidlayers_z_aux+[dim_z_aux,], activation)
            self.func_z_aux_lnvar = MLP([num_units_feat,]+hidlayers_z_aux+[dim_z_aux,], activation)

        if not no_phy:
            hidlayers_z_phy = config['arch']['phys_vae']['hidlayers_z_phy'] #[64, 32]
            # self.func_unmixer_coeff = nn.Sequential(MLP([num_units_feat,]+hidlayers_z_phy+[in_channels,], activation), nn.Tanh())
            self.func_z_phy_mean = nn.Sequential(MLP([num_units_feat,]+hidlayers_z_phy+[dim_z_phy,], activation), nn.Softplus()) #NOTE Softplus is used to ensure positive values
            self.func_z_phy_lnvar = MLP([num_units_feat,]+hidlayers_z_phy+[dim_z_phy,], activation) 

class Decoders(nn.Module):
    def __init__(self, config:dict):
        super(Decoders, self).__init__()

        in_channels = config['arch']['args']['input_dim'] #11 for RTM, 36 for Mogi
        dim_z_aux = config['arch']['phys_vae']['dim_z_aux'] #2
        dim_z_phy = config['arch']['phys_vae']['dim_z_phy'] #7 for RTM, 4 for Mogi
        activation = config['arch']['phys_vae']['activation'] #elu 
        no_phy = config['arch']['phys_vae']['no_phy']
        # x_lnvar = config['trainer']['phys_vae']['x_lnvar'] # default -9, does not make much sense
        
        # self.register_buffer('param_x_lnvar', torch.ones(1)*x_lnvar)

        # NOTE variable names can be simplified.
        if not no_phy:
            if dim_z_aux >= 0:
                # phy (and aux) -> x
                # TODO rename func_aux_expand to func_aux_res
                # self.func_aux_expand = MLP([dim_z_aux, 16, 32, 64, in_channels], activation)

                self.func_aux_dec = MLP([dim_z_aux, 16, 32, 64, in_channels], activation)
                self.func_correction = MLP([2 * in_channels, 4 * in_channels, in_channels], activation)#NOTE (optional) additional non-linear correction optional           

        else:
            # no phy
            self.func_aux_dec = MLP([dim_z_aux, 16, 32, 64, in_channels], activation) #TODO decide the decoding dimension, whether adding 16 dimension or not

class FeatureExtractor(nn.Module):
    def __init__(self, config:dict):
        super(FeatureExtractor, self).__init__()

        in_channels = config['arch']['args']['input_dim']#11 for RTM, 36 for Mogi
        hidlayers_feat = config['arch']['phys_vae']['hidlayers_feat']#[32,] NOTE change to 36 for Mogi to avoid information loss
        num_units_feat = config['arch']['phys_vae']['num_units_feat']#64
        activation = config['arch']['phys_vae']['activation']#elu
    
        # Shared backbone for feature extraction
        self.func_feat = MLP([in_channels,]+hidlayers_feat+[num_units_feat,], activation)


    def forward(self, x:torch.Tensor):
        return self.func_feat(x) #n*128 


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
        Rescale the output of the encoder to the physical parameters in the original scale
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
        Rescale the output of the encoder to the physical parameters in the original scale
        """
        z_phy_rescaled = {}
        for i, para_name in enumerate(self.z_phy_ranges.keys()):
            min = self.z_phy_ranges[para_name]['min']
            max = self.z_phy_ranges[para_name]['max']
            # if x shape is (batch, sequence, feature), then x[:,:,i] is the i-th feature
            if len(z_phy.shape) == 3:
                z_phy_rescaled[para_name] = z_phy[:, :, i]*(max-min)+min
            else:
                z_phy_rescaled[para_name] = z_phy[:, i]*(max-min)+min

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
        # self.physics_model = Physics(config) 
        self.physics_model = self.physics_init(config)
    
    def physics_init(self, config:dict):
        if config['arch']['args']['physics'] == 'RTM':
            return Physics_RTM(config)
        elif config['arch']['args']['physics'] == 'Mogi':
            return Physics_Mogi(config)
        else:
            raise ValueError("Unknown model type")

    def priors(self, n:int, device:torch.device):
        # prior_z_phy_mean = torch.cat([
        #     torch.ones(n,1,device=device) * 0.5 * (0 + 1) for i in range(self.dim_z_phy)
        # ], dim=1)
        # prior_z_phy_std = torch.cat([
        #     torch.ones(n,1,device=device) * max(1e-3, 0.866*(1 - 0)) for i in range(self.dim_z_phy)
        # ], dim=1)

        # NOTE KL term for z_phy can be removed later on
        prior_z_phy_mean = torch.full((n, self.dim_z_phy), 0.5 * (0+1), device=device)  # mean = 0.5
        prior_z_phy_std = torch.full((n, self.dim_z_phy), 0.866 * (1-0), device=device)   # std = 0.1

        prior_z_phy_stat = {'mean': prior_z_phy_mean, 'lnvar': 2.0*torch.log(prior_z_phy_std)}
        prior_z_aux_stat = {'mean': torch.zeros(n, max(0,self.dim_z_aux), device=device),
                            'lnvar': torch.zeros(n, max(0,self.dim_z_aux), device=device)}
        
        return prior_z_phy_stat, prior_z_aux_stat


    def generate_physonly(self, z_phy:torch.Tensor, const:dict=None):
        y = self.physics_model(z_phy, const=const) # (n, in_channels) 
        return y


    def decode(self, z_phy:torch.Tensor, z_aux:torch.Tensor, full:bool=False, const:dict=None):
        # NOTE this function can be simplified with more clear variable names. For now, it is kept similar to the original code.
        if not self.no_phy:
            # with physics
            y = self.physics_model(z_phy, const=const) # (n, in_channels)
            x_P = y
            if self.dim_z_aux >= 0:
                # expanded = self.dec.func_aux_expand(torch.cat([z_phy, z_aux], dim=1))
                correction = self.dec.func_aux_expand(z_aux) # (n, in_channels)
                x_PB = x_P + correction
                # expanded = self.dec.func_aux_dec(z_aux) # (n, in_channels)
                # x_PB = self.dec.func_correction(torch.cat([x_P, expanded], dim=1))
            else:
                x_PB = x_P.clone() # No correction if no auxiliary variables (dim_z_aux = -1)
        else:
            # no physics 
            y = torch.zeros(z_phy.shape[0], self.in_channels)
            if self.dim_z_aux >= 0:
                x_PB = self.dec.func_aux_dec(z_aux) 
            else:
               x_PB = torch.zeros(z_phy.shape[0], self.in_channels)
            x_P = x_PB.clone()

        if full:
            return x_PB, x_P, y
        else:
            return x_PB


    def encode(self, x:torch.Tensor):
        # x_ = x.view(-1, 3, 69, 69)
        x_ = x
        n = x_.shape[0]
        device = x_.device

        # infer z_aux
        feature_aux = self.enc.func_feat(x_)
        if self.dim_z_aux > 0:
            z_aux_stat = {'mean':self.enc.func_z_aux_mean(feature_aux), 'lnvar':self.enc.func_z_aux_lnvar(feature_aux)}
        else:
            z_aux_stat = {'mean':torch.empty(n, 0, device=device), 'lnvar':torch.empty(n, 0, device=device)}

        # infer z_phy
        if not self.no_phy:
            # coeff = self.enc.func_unmixer_coeff(feature_aux) # (n,11) for RTM, (n,36) for Mogi
            # unmixed = x_ * coeff # element-wise weighting for 1D data
            # feature_phy = self.enc.func_feat(unmixed)
            feature_phy = feature_aux
            z_phy_stat = {'mean': self.enc.func_z_phy_mean(feature_phy), 'lnvar': self.enc.func_z_phy_lnvar(feature_phy)}
        else:
            # unmixed = torch.zeros(n, self.in_channels, device=device)
            z_phy_stat = {'mean':torch.empty(n, 0, device=device), 'lnvar':torch.empty(n, 0, device=device)}

        # return z_phy_stat, z_aux_stat, unmixed
        return z_phy_stat, z_aux_stat


    def draw(self, z_phy_stat:dict, z_aux_stat:dict, hard_z:bool=False):
        if not hard_z:
            z_phy = draw_normal(z_phy_stat['mean'], z_phy_stat['lnvar'])
            z_aux = draw_normal(z_aux_stat['mean'], z_aux_stat['lnvar'])
        else:
            z_phy = z_phy_stat['mean'].clone()
            z_aux = z_aux_stat['mean'].clone()

        # cut infeasible regions
        if not self.no_phy:
            z_phy = torch.clamp(z_phy, 0, 1)

        return z_phy, z_aux


    def forward(self, x:torch.Tensor, reconstruct:bool=True, hard_z:bool=False,
                inference:bool=False, const:dict=None):
        # z_phy_stat, z_aux_stat, unmixed, = self.encode(x)
        z_phy_stat, z_aux_stat = self.encode(x)

        if not reconstruct:
            return z_phy_stat, z_aux_stat
        
        if not inference:
            # draw & reconstruction
            x_mean = self.decode(*self.draw(z_phy_stat, z_aux_stat, hard_z=hard_z), full=False, const=const)

            return z_phy_stat, z_aux_stat, x_mean
        else:
            z_phy, z_aux = self.draw(z_phy_stat, z_aux_stat, hard_z=hard_z)
            x_PB, x_P, _ = self.decode(z_phy, z_aux, full=True, const=const)
            return z_phy, z_aux, x_PB, x_P
