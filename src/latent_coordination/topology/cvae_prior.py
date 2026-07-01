import torch
import torch.nn as nn

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

class GeometryConditionedCVAEPrior(nn.Module):
    """Module D: Geometry-Conditioned CVAE Graph Prior"""
    def __init__(self, query_dim: int, geo_dim: int = 9, latent_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.query_dim = query_dim
        self.geo_dim = geo_dim
        self.latent_dim = latent_dim
        
        # Conditioned on x = [q || Geo_L]
        self.encoder = nn.Sequential(
            nn.Linear(query_dim + geo_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2) # mu, logvar
        )
        
    def forward(self, q: torch.Tensor, geo_l: torch.Tensor):
        # Concatenate query representation with geometric risk vector
        x = torch.cat([q, geo_l], dim=-1)
        params = self.encoder(x)
        mu, logvar = torch.chunk(params, 2, dim=-1)
        
        # Reparameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar
