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

class RecursiveLatentCore(nn.Module):
    """Module C: Recursive Latent Space Reasoning Core"""
    def __init__(self, hub_dim: int = 512, max_steps: int = 10, tau_exit: float = 0.8):
        super().__init__()
        self.w1 = nn.Linear(hub_dim, hub_dim)
        self.w2 = nn.Linear(hub_dim, hub_dim)
        self.act = nn.GELU()
        
        self.exit_classifier = nn.Linear(hub_dim, 1)
        self.max_steps = max_steps
        self.tau_exit = tau_exit
        
    def forward(self, z_init: torch.Tensor):
        z_t = z_init
        for t in range(self.max_steps):
            # Two-layer bottleneck residual network
            residual = self.w2(self.act(self.w1(z_t)))
            z_t = z_t + residual
            
            # Single-layer classifier with a sigmoid gate for early-exit
            exit_prob = torch.sigmoid(self.exit_classifier(z_t))
            if (exit_prob > self.tau_exit).all():
                break
                
        return z_t
