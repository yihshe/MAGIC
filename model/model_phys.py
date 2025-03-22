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
        dim_z_aux2 = config['arch']['phys_vae']['dim_z_aux2']#2
        dim_z_phy = config['arch']['phys_vae']['dim_z_phy']#7 for RTM, 4 for Mogi
        activation = config['arch']['phys_vae']['activation'] #TODO same as galaxy for gaussian sampling
        num_units_feat = config['arch']['phys_vae']['num_units_feat']#128
        
        self.func_feat = FeatureExtractor(config)

        if dim_z_aux2 > 0:
            hidlayers_aux2_enc = config['arch']['phys_vae']['hidlayers_aux2_enc'] #[64, 32]
            self.func_z_aux2_mean = MLP([num_units_feat,]+hidlayers_aux2_enc+[dim_z_aux2,], activation)
            self.func_z_aux2_lnvar = MLP([num_units_feat,]+hidlayers_aux2_enc+[dim_z_aux2,], activation)

        if not no_phy:
            hidlayers_z_phy = config['arch']['phys_vae']['hidlayers_z_phy'] #[64, 32]
            self.func_unmixer_coeff = nn.Sequential(MLP([num_units_feat,]+hidlayers_z_phy+[in_channels,], activation), nn.Tanh())
            self.func_z_phy_mean = nn.Sequential(MLP([num_units_feat,]+hidlayers_z_phy+[dim_z_phy,], activation), nn.Softplus()) 
            self.func_z_phy_lnvar = MLP([num_units_feat,]+hidlayers_z_phy+[dim_z_phy,], activation) 

class Decoders(nn.Module):
    def __init__(self, config:dict):
        super(Decoders, self).__init__()

        in_channels = config['arch']['args']['input_dim'] #11 for RTM, 36 for Mogi
        dim_z_aux2 = config['arch']['phys_vae']['dim_z_aux2'] #2
        dim_z_phy = config['arch']['phys_vae']['dim_z_phy'] #7 for RTM, 4 for Mogi
        activation = config['arch']['phys_vae']['activation'] #elu TODO relu or elu?
        no_phy = config['arch']['phys_vae']['no_phy']
        x_lnvar = config['trainer']['phys_vae']['x_lnvar'] #TODO not sure how its value is set exactly
        
        self.register_buffer('param_x_lnvar', torch.ones(1)*x_lnvar)

        if not no_phy:
            if dim_z_aux2 >= 0:
                # phy (and aux) -> x
                self.func_aux2_expand = MLP([dim_z_phy+dim_z_aux2, 32, 64, in_channels], activation)
                self.func_aux2_map = MLP([2 * in_channels, 4 * in_channels, in_channels], activation)            

        else:
            # no phy
            self.func_aux2_dec = MLP([dim_z_aux2, 16, 32, 64, in_channels], activation)

class FeatureExtractor(nn.Module):
    def __init__(self, config:dict):
        super(FeatureExtractor, self).__init__()

        in_channels = config['arch']['args']['input_dim']#11 for RTM, 36 for Mogi
        hidlayers_feat = config['arch']['phys_vae']['hidlayers_feat']#[64, 64]
        num_units_feat = config['arch']['phys_vae']['num_units_feat']#128
        activation = config['arch']['phys_vae']['activation']#TODO relu or elu?
    
        # Shared backbone for feature extraction)
        self.func_feat = MLP([in_channels,]+hidlayers_feat+[num_units_feat,], activation)


    def forward(self, x:torch.Tensor):
        return self.func_feat(x) #n*128 


class Physics_RTM(nn.Module):
    def __init__(self, config:dict):
        super(Physics_RTM, self).__init__()
        # TODO implement for Mogi
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
    
    def forward(self, z_phy:torch.Tensor):
        z_phy_rescaled = self.rescale(z_phy)
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
    
    def forward(self, z_phy:torch.Tensor):
        z_phy_rescaled = self.rescale(z_phy)
        output = self.model.run(**z_phy_rescaled)
        return (output - self.x_mean) / self.x_scale 

class PHYS_VAE(nn.Module):
    def __init__(self, config:dict):
        super(PHYS_VAE, self).__init__()

        self.no_phy = config['arch']['phys_vae']['no_phy']
        self.dim_z_aux2 = config['arch']['phys_vae']['dim_z_aux2']
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
        prior_z_phy_mean = torch.cat([
            torch.zeros(n,1,device=device) * 0.5 * (0 + 1) for i in range(self.dim_z_phy)
        ], dim=1)
        prior_z_phy_std = torch.cat([
            torch.ones(n,1,device=device) * max(1e-3, 0.866*(1 - 0)) for i in range(self.dim_z_phy)
        ], dim=1)

        prior_z_phy_stat = {'mean': prior_z_phy_mean, 'lnvar': 2.0*torch.log(prior_z_phy_std)}#TODO check lnvar and reparameterization
        prior_z_aux2_stat = {'mean': torch.zeros(n, max(0,self.dim_z_aux2), device=device),
            'lnvar': torch.zeros(n, max(0,self.dim_z_aux2), device=device)}#TODO why lnvar for z_aux2 is defined based on dim_z_aux2?
        return prior_z_phy_stat, prior_z_aux2_stat


    def generate_physonly(self, z_phy:torch.Tensor):
        y = self.physics_model(z_phy) # (n, in_channels) 
        return y


    def decode(self, z_phy:torch.Tensor, z_aux2:torch.Tensor, full:bool=False):
        if not self.no_phy:
            # with physics
            y = self.physics_model(z_phy) # (n, in_channels)
            # x_P = y.repeat(1,3,1,1)
            x_P = y
            if self.dim_z_aux2 >= 0:
                expanded = self.dec.func_aux2_expand(torch.cat([z_phy, z_aux2], dim=1))
                x_PB = self.dec.func_aux2_map(torch.cat([x_P, expanded], dim=1))
            else:
                x_PB = x_P.clone() # No correction if no auxiliary variables (dim_z_aux2 = -1)
        else:
            # no physics 
            y = torch.zeros(z_phy.shape[0], self.in_channels)
            if self.dim_z_aux2 >= 0:
                x_PB = self.dec.func_aux2_dec(z_aux2) 
            else:
               x_PB = torch.zeros(z_phy.shape[0], self.in_channels)
            x_P = x_PB.clone()

        if full:
            return x_PB, x_P, self.dec.param_x_lnvar, y
        else:
            return x_PB, self.dec.param_x_lnvar


    def encode(self, x:torch.Tensor):
        # x_ = x.view(-1, 3, 69, 69)
        x_ = x
        n = x_.shape[0]
        device = x_.device

        # infer z_aux2
        feature_aux2 = self.enc.func_feat(x_)
        if self.dim_z_aux2 > 0:
            z_aux2_stat = {'mean':self.enc.func_z_aux2_mean(feature_aux2), 'lnvar':self.enc.func_z_aux2_lnvar(feature_aux2)}
        else:
            z_aux2_stat = {'mean':torch.empty(n, 0, device=device), 'lnvar':torch.empty(n, 0, device=device)}

        # infer z_phy
        if not self.no_phy:
            coeff = self.enc.func_unmixer_coeff(feature_aux2) # (n,11) for RTM, (n,36) for Mogi
            unmixed = x_ * coeff # element-wise weighting for 1D data
            feature_phy = self.enc.func_feat(unmixed)
            z_phy_stat = {'mean': self.enc.func_z_phy_mean(feature_phy), 'lnvar': self.enc.func_z_phy_lnvar(feature_phy)}
        else:
            unmixed = torch.zeros(n, self.in_channels, device=device)
            z_phy_stat = {'mean':torch.empty(n, 0, device=device), 'lnvar':torch.empty(n, 0, device=device)}

        return z_phy_stat, z_aux2_stat, unmixed


    def draw(self, z_phy_stat:dict, z_aux2_stat:dict, hard_z:bool=False):
        if not hard_z:
            z_phy = draw_normal(z_phy_stat['mean'], z_phy_stat['lnvar'])#TODO check lnvar and reparameterization
            z_aux2 = draw_normal(z_aux2_stat['mean'], z_aux2_stat['lnvar'])
        else:
            z_phy = z_phy_stat['mean'].clone()
            z_aux2 = z_aux2_stat['mean'].clone()

        # cut infeasible regions
        if not self.no_phy:
            z_phy = torch.clamp(z_phy, 0, 1)

        return z_phy, z_aux2


    def forward(self, x:torch.Tensor, reconstruct:bool=True, hard_z:bool=False,
                inference:bool=False):
        z_phy_stat, z_aux2_stat, unmixed, = self.encode(x)

        if not reconstruct:
            return z_phy_stat, z_aux2_stat
        
        if not inference:
            # draw & reconstruction
            x_mean, x_lnvar = self.decode(*self.draw(z_phy_stat, z_aux2_stat, hard_z=hard_z), full=False)

            return z_phy_stat, z_aux2_stat, x_mean, x_lnvar
        else:
            z_phy, z_aux2 = self.draw(z_phy_stat, z_aux2_stat, hard_z=hard_z)
            x_PB, x_P, x_lnvar, y = self.decode(z_phy, z_aux2, full=True)
            return z_phy, z_aux2, x_PB, x_P
