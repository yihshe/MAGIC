import numpy as np
import torch

class DPM():
    """
    A class used to represent a EVI time series using the Mogi (1958) model.
    PyTorch implementation adapted from the original Versatile Modeling of 
    Deformation (VMOD) implementation. 

    Attributes
    ----------
    parameters : array
        names for the parameters in the model
    """
    def __init__(self, **kwargs):
        super(DPM, self).__init__()
        """
        Initialize the DPM with default parameters.

        Parameters:
            x (Tensor): x-coordinate of stations (1D tensor)
            y (Tensor): y-coordinate of stations (1D tensor)
            nu (float): poisson's ratio for medium (default 0.25)
            mu (float): shear modulus for medium (Pa) (default 4e9)
        """
        # TODO: add default parameters for DPM
        # self.x = x
        # self.y = y
        # self.nu = nu
    
    def run(self, xcen, ycen, d, dV):
        """
        Function to run the forward model of Mogi (1958) for batched data.
        3d displacement field on surface from point source (Mogi, 1958).

        Parameters:
            xcen (Tensor): x-offset of point source epicenter (m)
            ycen (Tensor): y-offset of point source epicenter (m)
            d (Tensor): depth to point (m)
            dV (Tensor): change in volume (m^3)
        
        Returns:
            ux (Tensor) : displacements in east in meters.
            uy (Tensor) : displacements in north in meters.
            uz (Tensor) : displacements in vertical in meters.
            output (Tensor) : concatenated displacements in east, north, vertical in millimeters.
        """
        # # Center coordinate grid on point source
        # x_adjusted = self.x - xcen.unsqueeze(1)
        # y_adjusted = self.y - ycen.unsqueeze(1)

        # # Convert to surface cylindrical coordinates
        # th, rho = self.cart2pol(x_adjusted, y_adjusted) # surface angle and radial distance

        # # Compute radial distance from source
        # R = torch.sqrt(d.unsqueeze(1)**2 + rho**2) # radial distance from source

        # # Mogi displacement calculation 
        # C = ((1-self.nu) / torch.pi) * dV.unsqueeze(1)
        # ur = C * rho / R**3 # horizontal displacement, m
        # uz = C * d.unsqueeze(1) / R**3 # vertical displacement, m

        # ux, uy = self.pol2cart(th, ur)

        # output = torch.cat((ux, uy, uz), dim=1)*1e3 # convert to mm

        # # concatenate the displacement fields horizontally
        # TODO physical parameters to invert
        output = None
        return output