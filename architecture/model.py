"""
Composes the fixed baseline architecture (baseline.py) with either a
standard activation or the experimental sparse activation (modules/kwta.py).

This is the only file that should need to change to add a new mechanism
to test — swap in a new activation factory here, keep baseline.py frozen.

Builders are ADDITIVE. `build_dense_model` and `build_kwta_model` are exactly as
they were for the v1 (commit 26ba704) and v2 (9e7874f) runs and must not change,
or those recorded results stop being reproducible.
"""

import torch.nn as nn

from architecture.baseline import SmallCNN
from architecture.modules.kwta import KWinnersTakeAll
from architecture.modules.kwta_v2 import KWinnersTakeAllV2


def build_dense_model(num_classes: int = 10) -> nn.Module:
    """Standard baseline: ReLU activations, fully dense."""
    return SmallCNN(num_classes=num_classes, activation_fn=nn.ReLU)


def build_kwta_model(num_classes: int = 10, k: float = 0.2) -> nn.Module:
    """
    Experimental model: same architecture as the baseline, but each ReLU
    is followed by a kWTA sparsification step.

    k=0.2 means only the top 20% most active units per layer stay on for
    any given input; the rest are zeroed. Combining ReLU -> kWTA means
    kWTA only ever selects among already-nonnegative activations, which
    keeps the "winners" interpretation sensible.
    """

    def sparse_activation_factory():
        return nn.Sequential(nn.ReLU(), KWinnersTakeAll(k=k))

    return SmallCNN(num_classes=num_classes, activation_fn=sparse_activation_factory)


def build_kwta_v2_model(num_classes: int = 10, k: float = 0.2,
                        dim: str = "channel", boost: float = 0.0) -> nn.Module:
    """
    Stage 1 experimental model: same architecture again, with the competition
    axis and duty-cycle boosting switchable so they can be ablated separately.

    Protocol v2 showed the v1 kWTA separates patterns as theorized (2.24x the
    dense baseline's same-vs-different-class code separation at 600 labels) but
    leaves 44-49% of units permanently dead, so it pays for those codes with half
    the network. `dim` and `boost` are the two candidate causes; see
    modules/kwta_v2.py.

    `dim="global", boost=0.0` reproduces `build_kwta_model` exactly, which makes
    it the anchor arm of the Stage 1 factorial.
    """

    def sparse_activation_factory():
        return nn.Sequential(nn.ReLU(), KWinnersTakeAllV2(k=k, dim=dim, boost=boost))

    return SmallCNN(num_classes=num_classes, activation_fn=sparse_activation_factory)
