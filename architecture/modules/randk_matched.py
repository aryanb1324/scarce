"""
Random-k, matched on SURVIVING signal instead of on nominal count.

WHAT THIS DOES
--------------
At each competition site -- every spatial position of a conv feature map, or
every sample of a 2-D layer -- it keeps k channels/units chosen uniformly at
random *from among the entries that are currently nonzero*. If fewer than k
nonzero entries exist at that site, it keeps all of them (and nothing else).
Values that survive pass through unchanged and unrescaled; everything else is
zeroed. The mask is applied in train() and eval() alike.

WHY IT EXISTS (this is a bug fix for a control, not a new mechanism)
--------------------------------------------------------------------
Stage 2 used `RandomKChannel` as the control for the only question left about
channel-wise kWTA: is the *input-dependent competition* doing the work, or would
any structured sparsification do? That control was designed to match kWTA on k,
on axis, and on exact surviving count. It matched the first two and not the
third, and the run's headline gap is confounded because of it.

Measured in the trained network, FC layer, nominal k = 0.2 x 128 = 26:

    arm               nominal selected     actually nonzero
    kwta_channel                    26                 26.0
    randk_channel                   26                  7.6

The cause is the post-ReLU zero fraction (~70% in the real network).
`topk` by value lands on the largest entries, which are all nonzero, so kWTA's
effective count equals its nominal count. Uniform random selection lands on
zeros in proportion to how many there are, so it discards most of its own
budget. `randk_channel` is therefore not "same sparsity, random selection" -- it
is "3.4x sparser AND random selection AND unrescaled," and the -11 pt collapse
it produced mixes at least three causes.

Restricting the random draw to the nonzero entries removes the mismatch by
construction rather than by tuning k to compensate:

    kWTA at a site keeps the k largest values. After ReLU every nonzero is
    strictly greater than every zero, so kWTA keeps min(k, n_nonzero) live
    units and pads with zeros when it runs out.

    This module keeps k of the nonzeros, i.e. also exactly min(k, n_nonzero)
    live units.

The two arms then differ in *exactly one* property -- whether the surviving
subset is chosen by magnitude or at random -- which is what the control was
supposed to isolate all along.

WHAT IT CONTROLS FOR, AND WHAT IT DOES NOT
------------------------------------------
Controls for: nominal k, competition axis, number of live units per site,
total surviving unit count, always-on train/eval behaviour, absence of
rescaling, tensor shapes, parameter count.

Does NOT control for: the *magnitude* of the surviving activations. kWTA keeps
the largest values, so it necessarily carries more activation mass than a random
subset of the same size. That is not a defect of the control -- "the winners are
the big ones" IS the competition, and equalizing mass too would equalize away
the thing under test. Unit count is the confound; mass is the mechanism.

NEUROSCIENCE FRAMING (deliberately none)
----------------------------------------
Unlike the kWTA modules, this one has no biological motivation and makes no
data-efficiency prediction of its own. It is a null arm whose entire purpose is
to make the kWTA arm's claim falsifiable. If accuracy is indistinguishable
between the two, the honest description of the effect is "structured sparsity at
the right level," and the lateral-inhibition framing should be dropped rather
than defended.

IMPLEMENTATION NOTE
-------------------
Selection is `topk` over a score built as `uniform_noise + 1[x != 0]`: nonzero
entries score in [1, 2), zeros in [0, 1), so every nonzero outranks every zero
and the ordering *within* the nonzeros is uniform noise. This yields exactly k
selected slots per site (matching kWTA's mask shape exactly) while guaranteeing
zeros are chosen only after the nonzeros are exhausted -- at which point the
choice is irrelevant, because masking a zero changes nothing. No sampling loop,
no ragged tensors, one kernel.

Caveat: "live" is defined as `x != 0`, so a *negative* input counts as live
whereas kWTA's `topk` would rank it below a zero. Both modules are placed
directly after a ReLU in `model.py`, where negatives cannot occur, so the two
definitions coincide in every use in this project. Do not reuse this module
before a signed activation without revisiting that.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RandomKMatchedChannel(nn.Module):
    """
    Keeps k randomly-chosen NONZERO channels at each spatial position (4-D input)
    or k randomly-chosen nonzero units per sample (2-D input). Keeps all of them
    when fewer than k are nonzero.

    Interface-compatible with `RandomKChannel`; the only difference is that the
    random draw is restricted to the entries that survived the preceding
    activation, which makes the live-unit count identical to
    `KWinnersTakeAllV2(dim="channel")` on the same input rather than merely
    nominally equal.

    Args:
        k: units to keep per site. int = exact count, float in (0, 1] = fraction
            of the channel/feature dimension. Resolved exactly as in
            `KWinnersTakeAll._resolve_k`, so the two arms round identically.
    """

    def __init__(self, k: float | int):
        super().__init__()
        self.k = k

    def _resolve_k(self, num_units: int) -> int:
        if isinstance(self.k, float):
            return max(1, int(round(self.k * num_units)))
        return min(self.k, num_units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = 1
        n_units = x.shape[dim]
        k = self._resolve_k(n_units)
        with torch.no_grad():
            # Nonzeros score in [1, 2), zeros in [0, 1): every live entry
            # outranks every dead one, and live entries are ordered by pure
            # noise. topk therefore draws k uniformly from the live set, and
            # only spills onto zeros once the live set is exhausted.
            score = torch.rand_like(x) + (x != 0).to(x.dtype)
            idx = score.topk(k, dim=dim).indices
        mask = torch.zeros_like(x).scatter_(dim, idx, 1.0)
        return x * mask

    def extra_repr(self) -> str:
        return f"k={self.k}, selection=random-among-nonzero"
