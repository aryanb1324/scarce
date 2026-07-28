"""
k-Winner-Take-All, v2: configurable competition axis + duty-cycle boosting.

WHY THIS EXISTS (evidence, not speculation)
-------------------------------------------
Protocol v2 (commit 9e7874f) measured two things about the v1 kWTA module:

  * It WORKS as theorized. Pairwise Jaccard separation between same-class and
    different-class codes was 2.24x the dense baseline's at 600 labels, 1.21x at
    3,000, 1.07x at 6,000, 0.99x at 30,000, 0.88x at 48,000 -- monotonically
    largest exactly where data is scarcest, which is the hypothesis's own
    mechanistic prediction.

  * It is also HALF DEAD. 44-49% of units never won kWTA across a full
    validation pass, at every budget. A unit that never wins receives no
    gradient, so its weights never move, so it keeps losing. kWTA is buying its
    better codes with roughly half the network -- which is why the separation
    advantage never converted into accuracy (paired delta -1.58 pts at 600
    labels, n.s.).

So the mechanism is not the problem; the implementation is spending capacity it
does not need to spend. This module fixes the two candidate causes, each
independently switchable so the next experiment can attribute the effect.

CAUSE 1 -- the competition axis (`dim`)
---------------------------------------
v1 flattens (B, C, H, W) to (B, C*H*W) and takes ONE global top-k. For conv1
that is a competition among 12,544 units spanning channels and space at once, so
a whole channel can lose everywhere and die. It is also not lateral inhibition:
because a digit occupies a small part of the frame, the survivors are determined
mostly by WHERE the strokes are -- spatial saliency masking.

Cortical lateral inhibition competes among feature detectors sharing a receptive
field: top-k across channels at each spatial position (16 units, not 12,544).
Under that rule exactly k channels win at EVERY position by construction, so a
weak channel still wins wherever it is locally best, and dead channels become
structurally much harder to produce. The axis may therefore fix the dead units on
its own -- which is a testable prediction, not an assumption.

  dim="channel" : top-k across C at each (h, w)   <- the actual hypothesis
  dim="global"  : v1 behaviour, kept as the ablation arm

For a 2-D (linear-layer) input both rules are identical, so `dim` only changes
the conv blocks -- a clean isolated comparison.

Importantly the two axes keep the SAME total sparsity: channel-wise retains
k_c = k*C channels at each of H*W positions, i.e. k*C*H*W units, which is exactly
what the global rule retains. The axis change is a pure redistribution of where
the surviving activity sits, not a change in how much survives, so an accuracy
difference between the arms cannot be explained by one of them simply being less
sparse.

CAUSE 2 -- no duty-cycle boosting (`boost`)
-------------------------------------------
A hard top-k with no compensation is a ratchet: early losers never recover. The
v1 docstring advertised a `boost_off` argument that was never implemented.

Boosting tracks each unit's running win rate and inflates the RANKING score of
chronically-losing units so they re-enter the competition:

    duty_i  <- (1 - alpha) * duty_i + alpha * win_rate_i        (train mode only)
    boost_i =  exp(clamp(beta * (1 - duty_i / target), +-3))     target = k / n
    score   =  x * boost          -> rank by score, but pass through raw x

A unit at its target rate gets boost 1; a fully dead unit gets exp(beta); an
over-firing unit is damped. The normalized form makes beta scale-free across
layers of very different widths. Boosting affects ranking ONLY -- the values that
pass through are the unmasked originals -- and is disabled in eval mode, so the
learned representation is not perturbed at inference.

NOTE (non-standard training behaviour, per architecture/skills.md): with
boost > 0 this module behaves differently in train() and eval() mode, like
BatchNorm and unlike v1 kWTA. It also carries a `duty` buffer that is part of
`state_dict`. Both matter for checkpointing and for anything that assumes
train/eval parity.

BACKWARD COMPATIBILITY: with dim="global", boost=0.0 this is numerically
IDENTICAL to v1's KWinnersTakeAll -- asserted in tests/test_kwta_v2.py, so the
v2 results remain the anchor arm of the next experiment.
"""

import torch

from architecture.modules.kwta import KWinnersTakeAll


class KWinnersTakeAllV2(KWinnersTakeAll):
    """
    Args:
        k: units to keep. int = exact count, float in (0,1] = fraction.
        dim: "channel" (top-k across channels per spatial position) or "global"
            (v1: top-k over the flattened channel x space tensor). Ignored for
            2-D inputs, where both are the same operation.
        boost: duty-cycle boost strength (beta). 0.0 disables boosting entirely
            and restores exact v1 behaviour.
        duty_alpha: EMA rate for the running win-rate estimate.
        max_boost_exp: clamp on the boost exponent, so a long-dead unit cannot
            produce an unbounded multiplier.
    """

    def __init__(self, k, dim: str = "channel", boost: float = 0.0,
                 duty_alpha: float = 0.01, max_boost_exp: float = 3.0):
        super().__init__(k)
        if dim not in ("channel", "global"):
            raise ValueError(f"dim must be 'channel' or 'global', got {dim!r}")
        self.dim = dim
        self.boost = float(boost)
        self.duty_alpha = float(duty_alpha)
        self.max_boost_exp = float(max_boost_exp)
        # Lazily sized on first use: the activation factory in model.py does not
        # know the layer width, and baseline.py is frozen so it cannot pass one.
        self.register_buffer("duty", torch.zeros(0))

    # -- internals ----------------------------------------------------------

    def _ensure_duty(self, n: int, target: float, ref: torch.Tensor) -> None:
        if self.duty.numel() != n:
            # Initialize AT target so the boost factor starts at exactly 1.0 and
            # early training is unperturbed.
            self.duty = torch.full((n,), float(target),
                                   device=ref.device, dtype=ref.dtype)

    def _boost_factor(self, n: int, target: float, ref: torch.Tensor) -> torch.Tensor:
        self._ensure_duty(n, target, ref)
        e = (self.boost * (1.0 - self.duty / target)).clamp(
            -self.max_boost_exp, self.max_boost_exp)
        return torch.exp(e)

    def _update_duty(self, win_rate: torch.Tensor, target: float) -> None:
        self._ensure_duty(win_rate.numel(), target, win_rate)
        self.duty.mul_(1.0 - self.duty_alpha).add_(self.duty_alpha * win_rate.detach())

    @property
    def _boosting(self) -> bool:
        return self.training and self.boost > 0.0

    # -- forward ------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and self.dim == "channel":
            n_units = x.shape[1]
            k = self._resolve_k(n_units)
            target = k / n_units
            with torch.no_grad():
                score = x
                if self._boosting:
                    score = x * self._boost_factor(n_units, target, x).view(1, -1, 1, 1)
                idx = score.topk(k, dim=1).indices
            mask = torch.zeros_like(x).scatter_(1, idx, 1.0)
            if self._boosting:
                self._update_duty(mask.mean(dim=(0, 2, 3)), target)
            return x * mask

        # Global: flatten everything after the batch dim. Identical to v1 when
        # boost == 0, which is what makes the v2 run a valid anchor arm.
        shape = x.shape
        flat = x.reshape(shape[0], -1)
        n_units = flat.shape[1]
        k = self._resolve_k(n_units)
        target = k / n_units
        with torch.no_grad():
            score = flat
            if self._boosting:
                score = flat * self._boost_factor(n_units, target, flat).view(1, -1)
            idx = score.topk(k, dim=1).indices
        mask = torch.zeros_like(flat).scatter_(1, idx, 1.0)
        if self._boosting:
            self._update_duty(mask.mean(dim=0), target)
        return (flat * mask).view(shape)

    def extra_repr(self) -> str:
        return f"k={self.k}, dim={self.dim}, boost={self.boost}"
