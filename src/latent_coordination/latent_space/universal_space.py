import torch
import torch.nn as nn
from typing import Dict, Tuple

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

def cka_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_c = x - x.mean(dim=0, keepdim=True)
    y_c = y - y.mean(dim=0, keepdim=True)
    
    gram_x = torch.mm(x_c, x_c.t())
    gram_y = torch.mm(y_c, y_c.t())
    
    dot_xy = torch.sum(gram_x * gram_y)
    norm_x = torch.sqrt(torch.sum(gram_x * gram_x))
    norm_y = torch.sqrt(torch.sum(gram_y * gram_y))
    
    cka = dot_xy / (norm_x * norm_y + 1e-8)
    return 1.0 - cka

class AdapterPair(nn.Module):
    def __init__(self, input_dim: int, hub_dim: int = 512):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hub_dim)
        self.decoder = nn.Linear(hub_dim, input_dim)
    
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(h)
        h_rec = self.decoder(z)
        return z, h_rec

class UniversalLatentHub(nn.Module):
    def __init__(self, hub_dim: int = 512, lambda_dae: float = 1.0, lambda_cka: float = 1.0):
        super().__init__()
        self.hub_dim = hub_dim
        self.lambda_dae = lambda_dae
        self.lambda_cka = lambda_cka
        self.adapters = nn.ModuleDict()
        
    def add_language_adapter(self, lang: str, input_dim: int):
        self.adapters[lang] = AdapterPair(input_dim, self.hub_dim)
        
    def fit_isolated_adapter(self, target_lang: str, input_dim: int, train_loader, optimizer, noise_std: float = 0.1, anchor_lang: str = "en"):
        """Adapter Scaling Protocol: Fit an isolated adapter to the frozen hub."""
        if target_lang not in self.adapters:
            self.add_language_adapter(target_lang, input_dim)
            
        adapter = self.adapters[target_lang]
        adapter.train()
        
        for batch in train_loader:
            optimizer.zero_grad()
            h_i = batch['hidden_states']
            
            # Reconstruction Loss
            z_i, h_rec = adapter(h_i)
            l_recon = torch.nn.functional.mse_loss(h_rec, h_i)
            
            # DAE Loss
            noise = torch.randn_like(h_i) * noise_std
            _, h_rec_noisy = adapter(h_i + noise)
            l_dae = torch.nn.functional.mse_loss(h_rec_noisy, h_i)
            
            # CKA Alignment with frozen anchor (e.g. English)
            if anchor_lang not in self.adapters:
                raise ValueError(f"Anchor language '{anchor_lang}' adapter missing. Cannot compute CKA alignment.")
            if 'anchor_hidden_states' not in batch:
                raise ValueError("Batch missing 'anchor_hidden_states'. Dummy 0.0 loss fallbacks are prohibited.")
                
            with torch.no_grad():
                z_anchor, _ = self.adapters[anchor_lang](batch['anchor_hidden_states'])
            
            l_cka = cka_loss(z_i, z_anchor)
            
            loss = l_recon + self.lambda_dae * l_dae + self.lambda_cka * l_cka
            loss.backward()
            optimizer.step()
