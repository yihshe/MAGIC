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

# xc = 0.0
# yc = 0.0

#                    I0   A      e      theta #NOTE feasible range
# feasible_range_lb = [0.1, 0.001, 0.0,   0]
# feasible_range_ub = [1.0, 10.0,  0.999, 3.142]


class Decoders(nn.Module):
    def __init__(self, config:dict):
        super(Decoders, self).__init__()

        dim_z_aux2 = config['arch']['phys_vae']['dim_z_aux2']
        activation = config['arch']['phys_vae']['activation'] #elu
        no_phy = config['arch']['phys_vae']['no_phy']
        x_lnvar = config['trainer']['phys_vae']['x_lnvar'] #TODO not sure how its value is set exactly

        self.register_buffer('param_x_lnvar', torch.ones(1)*x_lnvar)

        if not no_phy:
            if dim_z_aux2 >= 0:
                # unet_size = config['unet_size']

                # phy (and aux)
                k = 1
                self.func_aux2_expand1 = MLP([4+dim_z_aux2, 32, 64, 128], activation)
                self.func_aux2_expand2 = MLP([128, 64, len(self.rtm_paras)], self.activation)
                # self.func_aux2_expand2 = nn.Sequential(
                #     nn.ConvTranspose2d(128, k*16, 4, 1, 0, bias=False), nn.BatchNorm2d(k*16), nn.ReLU(True),
                #     nn.ConvTranspose2d(k*16, k*8, 4, 2, 1, bias=False), nn.BatchNorm2d(k*8), nn.ReLU(True),
                #     nn.ConvTranspose2d(k*8, k*4, 4, 2, 1, bias=False), nn.BatchNorm2d(k*4), nn.ReLU(True),
                #     nn.ConvTranspose2d(k*4, k*2, 5, 2, 1, bias=False), nn.BatchNorm2d(k*2), nn.ReLU(True),
                #     nn.ConvTranspose2d(k*2, 1, 5, 2, 0, bias=False),
                # )# NOTE for 1D data, func_aux2_expand1 should be sufficient
                # k = 4
                # self.func_aux2_map = UNet(unet_size)
                self.func_aux2_map = nn.Sequential(
                    nn.Linear(2 * len(self.bands_index), 4 * len(self.bands_index)),
                    nn.ReLU(),
                    nn.Linear(4 * len(self.bands_index), len(self.bands_index)),
                    nn.Sigmoid(),
                ) #TODO fine-tuned design, sigmoid or not              

        else:
            # no phy
            # k = 4
            # self.func_aux2_dec = nn.Sequential(
            #     nn.ConvTranspose2d(dim_z_aux2, k*32, 4, 1, 0, bias=False), nn.BatchNorm2d(k*32), nn.ReLU(True),
            #     nn.ConvTranspose2d(k*32, k*16, 4, 2, 1, bias=False), nn.BatchNorm2d(k*16), nn.ReLU(True),
            #     nn.ConvTranspose2d(k*16, k*8, 4, 2, 1, bias=False), nn.BatchNorm2d(k*8), nn.ReLU(True),
            #     nn.ConvTranspose2d(k*8, k*4, 5, 2, 1, bias=False), nn.BatchNorm2d(k*4), nn.ReLU(True),
            #     nn.ConvTranspose2d(k*4, 3, 5, 2, 0, bias=False),
            # )
            # TODO directly decoding auxiliary variables (dim_z_aux2) if no physics
            pass


class Encoders(nn.Module):
    def __init__(self, config:dict):
        super(Encoders, self).__init__()

        no_phy = config['arch']['phys_vae']['no_phy']
        num_units_feat = config['arch']['phys_vae']['num_units_feat']
        dim_z_aux2 = config['arch']['phys_vae']['dim_z_aux2']
        dim_z_phy = config['arch']['phys_vae']['dim_z_phy']
        activation = config['arch']['phys_vae']['activation'] #TODO same as galaxy for gaussian sampling
        # TODO until here
        self.func_feat = FeatureExtractor(config)

        if dim_z_aux2 > 0:
            hidlayers_aux2_enc = config['arch']['phys_vae']['hidlayers_aux2_enc']#TODO a list
            self.func_z_aux2_mean = MLP([num_units_feat,]+hidlayers_aux2_enc+[dim_z_aux2,], activation)
            self.func_z_aux2_lnvar = MLP([num_units_feat,]+hidlayers_aux2_enc+[dim_z_aux2,], activation)

        if not no_phy:
            hidlayers_z_phy = config['arch']['phys_vae']['hidlayers_z_phy'] #2, otherwise 2+4, depending on no_phy or not
            # self.func_unmixer_coeff = nn.Sequential(MLP([num_units_feat, 16, 16, 4], activation), nn.Tanh())
            self.func_unmixer_coeff = MLP([num_units_feat, np.round(num_units_feat*0.5), dim_z_phy], activation)#TODO activation should be tahn or sigmoid
            self.func_z_phy_mean = nn.Sequential(MLP([num_units_feat,]+hidlayers_z_phy+[dim_z_phy,], activation), nn.Softplus())#TODO learn a unit gaussian or not
            self.func_z_phy_lnvar = MLP([num_units_feat,]+hidlayers_z_phy+[dim_z_phy,], activation) #TODO activation for VAE?


class FeatureExtractor(nn.Module):
    def __init__(self, config:dict):
        super(FeatureExtractor, self).__init__()

        in_channels = config['arch']['input_dim']
        hidlayers_feat = config['arch']['phys_vae']['hidlayers_feat']
        num_units_feat = config['arch']['phys_vae']['num_units_feat']
        activation = config['arch']['phys_vae']['activation']
        # k = 8
        # self.convnet = nn.Sequential(
        #     nn.Conv2d(in_channels, k*2, 5, 1, 2), nn.BatchNorm2d(k*2), nn.LeakyReLU(0.2, inplace=True),
        #     nn.MaxPool2d(2), # 2k x34x34
        #     nn.Conv2d(k*2, k*4, 5, 1, 2), nn.BatchNorm2d(k*4), nn.LeakyReLU(0.2, inplace=True),
        #     nn.MaxPool2d(2), # 4k x17x17
        #     nn.Conv2d(k*4, k*8, 5, 1, 2), nn.BatchNorm2d(k*8), nn.LeakyReLU(0.2, inplace=True),
        #     nn.MaxPool2d(2), # 8k x8x8
        #     nn.Conv2d(k*8, k*16, 5, 1, 2), nn.BatchNorm2d(k*16), nn.LeakyReLU(0.2, inplace=True),
        #     nn.MaxPool2d(2), # 16k x4x4
        # )
        # self.after = nn.Linear(k*16*16, num_units_feat, bias=False)
        # Encoder (shared backbone for feature extraction)
        self.func_feat = MLP([in_channels,]+hidlayers_feat+[num_units_feat,], activation, actfun_output=True) # TODO activation of the last layer, and num_units_feat (128 or 512?)


    def forward(self, x:torch.Tensor):
        # x_ = x.view(-1, self.in_channels, 69, 69)
        # n = x_.shape[0]
        # return self.after(self.convnet(x_).view(n,-1))
        return self.func_feat(x) #TODO check the shape, n*128


# class Physics(nn.Module):
#     def __init__(self):
#         super(Physics, self).__init__()
#         self.register_buffer('x_coord', torch.linspace(-1.0, 1.0, 69).unsqueeze(0).repeat(69,1))
#         self.register_buffer('y_coord', torch.flipud(torch.linspace(-1.0, 1.0, 69).unsqueeze(1).repeat(1,69)))

#     def forward(self, z_phy:torch.Tensor):
#         # z_phy = [I0, A, e, theta]
#         n = z_phy.shape[0]

#         I0 = z_phy[:, 0].view(n,1,1)
#         A = z_phy[:, 1].view(n,1,1)
#         e = z_phy[:, 2].view(n,1,1)
#         B = A * (1.0 - e)
#         theta = z_phy[:, 3].view(n,1,1)

#         x = self.x_coord.unsqueeze(0).expand(n,69,69)
#         y = self.y_coord.unsqueeze(0).expand(n,69,69)
#         x_rotated = torch.cos(theta)*(x-xc) - torch.sin(theta)*(y-yc)
#         y_rotated = torch.sin(theta)*(x-xc) + torch.cos(theta)*(y-yc)
#         xx = x_rotated / A
#         yy = y_rotated / B
#         # r = torch.sqrt((x_rotated/A).pow(2) + (y_rotated/B).pow(2))
#         r = torch.norm(torch.cat([xx.unsqueeze(0), yy.unsqueeze(0)], dim=0), dim=0) # (n,69,69)

#         out = I0 * torch.exp(-r)
#         return out.unsqueeze(1) # (n,1,69,69) is returned; not (n,3,69,69)


class PHYS_VAE_RTM(nn.Module):
    def __init__(self, config:dict):
        super(PHYS_VAE_RTM, self).__init__()

        # assert config['range_I0'][0] <= config['range_I0'][1]
        # assert config['range_A'][0] <= config['range_A'][1]
        # assert config['range_e'][0] <= config['range_e'][1]
        # assert config['range_theta'][0] <= config['range_theta'][1]

        self.no_phy = config['arch']['phys_vae']['no_phy']
        self.dim_z_aux2 = config['arch']['phys_vae']['dim_z_aux2']
        self.activation = config['arch']['phys_vae']['activation']
        self.in_channels = config['arch']['input_dim']

        # self.range_I0 = config['range_I0']
        # self.range_A = config['range_A']
        # self.range_e = config['range_e']
        # self.range_theta = config['range_theta']

        # Decoding part
        self.dec = Decoders(config)

        # Encoding part
        self.enc = Encoders(config)

        # Physics
        # self.physics_model = Physics()
        self.physics_model = RTM() # TODO check the output shape. Implement also for Mogi

        # self.register_buffer('feasible_range_lb', torch.Tensor(feasible_range_lb))
        # self.register_buffer('feasible_range_ub', torch.Tensor(feasible_range_ub))


    def f(self, n:int, device:torch.device):
        # TODO adapt for RTM
        prior_z_phy_mean = torch.cat([
            torch.ones(n,1,device=device) * 0.5 * (self.range_I0[0] + self.range_I0[1]),
            torch.ones(n,1,device=device) * 0.5 * (self.range_A[0] + self.range_A[1]),
            torch.ones(n,1,device=device) * 0.5 * (self.range_e[0] + self.range_e[1]),
            torch.ones(n,1,device=device) * 0.5 * (self.range_theta[0] + self.range_theta[1]),
        ], dim=1)
        prior_z_phy_std = torch.cat([
            torch.ones(n,1,device=device) * max(1e-3, 0.866*(self.range_I0[1] - self.range_I0[0])),
            torch.ones(n,1,device=device) * max(1e-3, 0.866*(self.range_A[1] - self.range_A[0])),
            torch.ones(n,1,device=device) * max(1e-3, 0.866*(self.range_e[1] - self.range_e[0])),
            torch.ones(n,1,device=device) * max(1e-3, 0.866*(self.range_theta[1] - self.range_theta[0])),
        ], dim=1)
        prior_z_phy_stat = {'mean': prior_z_phy_mean, 'lnvar': 2.0*torch.log(prior_z_phy_std)}
        prior_z_aux2_stat = {'mean': torch.zeros(n, max(0,self.dim_z_aux2), device=device),
            'lnvar': torch.zeros(n, max(0,self.dim_z_aux2), device=device)}
        return prior_z_phy_stat, prior_z_aux2_stat


    def generate_physonly(self, z_phy:torch.Tensor):
        y = self.physics_model(z_phy) # (n,1,69,69) #TODO check the output of RTM 
        return y


    def decode(self, z_phy:torch.Tensor, z_aux2:torch.Tensor, full:bool=False):
        if not self.no_phy:
            # with physics
            y = self.physics_model(z_phy) # TODO reconstruction, normalized or not
            # x_P = y.repeat(1,3,1,1)
            x_P = y
            if self.dim_z_aux2 >= 0:
                expanded1 = self.dec.func_aux2_expand1(torch.cat([z_phy, z_aux2], dim=1)) # Concatenation of z_phy and z_aux2 along the channel dimension
                expanded2 = self.dec.func_aux2_expand2(expanded1)
                # x_PB = torch.sigmoid(self.dec.func_aux2_map(torch.cat([x_P, expanded2], dim=1)))
                x_PB = self.dec.func_aux2_map(torch.cat([x_P, expanded2], dim=1))
            else:
                x_PB = x_P.clone() # No correction if no auxiliary variables
        else:
            # no physics TODO until here 
            # y = torch.zeros(z_phy.shape[0], 1, 69, 69)
            y = torch.zeros(z_phy.shape[0], self.in_channels)
            if self.dim_z_aux2 >= 0:
                # x_PB = torch.sigmoid(self.dec.func_aux2_dec(z_aux2.view(-1, self.dim_z_aux2, 1, 1)))
                x_PB = self.dec.func_aux2_dec(z_aux2) #TODO change after deciding on func_aux2 the decoder
            else:
                # x_PB = torch.zeros(z_phy.shape[0], 3, 69, 69)
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
            # TODO here unmixed means separating the physics only part from the original input
            coeff = self.enc.func_unmixer_coeff(feature_aux2) # (n,3)
            # unmixed = torch.sum(x_*coeff.unsqueeze(2).unsqueeze(3), dim=1, keepdim=True)
            unmixed = x_ * coeff # element-wise weighting for 1D data
            feature_phy = self.enc.func_feat(unmixed)
            z_phy_stat = {'mean': self.enc.func_z_phy_mean(feature_phy), 'lnvar': self.enc.func_z_phy_lnvar(feature_phy)}
        else:
            # unmixed = torch.zeros(n, 3, 0, 0, device=device)
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
            n = z_phy.shape[0]
            # TODO replace with RTM ranges, and use unit gaussian or not 
            z_phy = torch.max(self.feasible_range_lb.unsqueeze(0).expand(n,4), z_phy)
            z_phy = torch.min(self.feasible_range_ub.unsqueeze(0).expand(n,4), z_phy)

        return z_phy, z_aux2


    def forward(self, x:torch.Tensor, reconstruct:bool=True, hard_z:bool=False):
        z_phy_stat, z_aux2_stat, unmixed, = self.encode(x)

        if not reconstruct:
            return z_phy_stat, z_aux2_stat

        # draw & reconstruction
        x_mean, x_lnvar = self.decode(*self.draw(z_phy_stat, z_aux2_stat, hard_z=hard_z), full=False)

        return z_phy_stat, z_aux2_stat, x_mean, x_lnvar
