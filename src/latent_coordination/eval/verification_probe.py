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

class LatentDriftException(Exception):
    """Raised when semantic drift and language confusion is detected inside the continuous hub."""
    pass

class QueryReconstructionProbe(nn.Module):
    """Module E: Closed-Loop Test-Time Reconstruction Probe"""
    def __init__(self, hub_dim: int = 512, query_dim: int = 1024, tau_drift: float = 0.5):
        super().__init__()
        self.decoder = nn.Linear(hub_dim, query_dim)
        self.tau_drift = tau_drift
        
    def forward(self, z_t: torch.Tensor, q_orig: torch.Tensor):
        q_rec = self.decoder(z_t)
        
        # Test-Time Fidelity Drift Score
        cos_sim = torch.nn.functional.cosine_similarity(q_rec, q_orig, dim=-1)
        drift_score = 1.0 - cos_sim
        
        # Error Mitigation
        if (drift_score > self.tau_drift).any():
            raise LatentDriftException("Semantic drift detected: Drift score exceeded safety threshold tau_drift.")
            
        return drift_score
