import logging

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

logger = logging.getLogger(__name__)


class RecursiveLatentCore(nn.Module):
    """Module C: Recursive Latent Space Reasoning Core.

    Runs a T-step two-layer bottleneck residual update
    ``z_t = z_{t-1} + W_2(GeLU(W_1(z_{t-1})))`` in the universal hub space with
    a sigmoid early-exit classifier (strategy.md §4.3, training-free
    LatentMAS-style recurrence). The number of steps actually used per call is
    exposed via :attr:`last_n_steps` and accumulated in :attr:`total_steps` /
    :attr:`total_calls` so the pipeline can log mean reasoning depth alongside
    wall-clock latency (the efficiency narrative requires both).
    """

    def __init__(
        self,
        hub_dim: int = 512,
        max_steps: int = 10,
        tau_exit: float = 0.8,
        zero_init_residual: bool = True,
    ):
        super().__init__()
        self.w1 = nn.Linear(hub_dim, hub_dim)
        self.w2 = nn.Linear(hub_dim, hub_dim)
        self.act = nn.GELU()
        if zero_init_residual:
            # Zero-init the residual output layer so an UNTRAINED core is an
            # exact identity map: wiring it into the live execution path then
            # measures the plumbing without perturbing hub states with random
            # residuals (the zero-mock policy forbids random-weight modules
            # silently mutating benchmark-path activations). Training later
            # moves w2 off zero and the recurrence becomes substantive.
            nn.init.zeros_(self.w2.weight)
            nn.init.zeros_(self.w2.bias)

        self.exit_classifier = nn.Linear(hub_dim, 1)
        self.max_steps = max_steps
        self.tau_exit = tau_exit

        # Per-call / cumulative step accounting (strategy.md §4.3: log mean T).
        self.last_n_steps: int = 0
        self.total_steps: int = 0
        self.total_calls: int = 0

    def forward(self, z_init: torch.Tensor) -> torch.Tensor:
        z_t = z_init
        n_steps = 0
        for t in range(self.max_steps):
            # Two-layer bottleneck residual network
            residual = self.w2(self.act(self.w1(z_t)))
            z_t = z_t + residual
            n_steps = t + 1

            # Single-layer classifier with a sigmoid gate for early-exit
            exit_prob = torch.sigmoid(self.exit_classifier(z_t))
            if (exit_prob > self.tau_exit).all():
                break

        self.last_n_steps = n_steps
        self.total_steps += n_steps
        self.total_calls += 1
        return z_t

    @property
    def mean_steps(self) -> float:
        """Mean reasoning steps per call (with early-exit active)."""
        return self.total_steps / self.total_calls if self.total_calls else 0.0
