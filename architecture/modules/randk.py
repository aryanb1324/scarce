"""
Random-k: the control that decides whether the competition is doing anything.

WHY THIS EXISTS
---------------
Stage 1b measured what channel-wise kWTA actually buys, and broke the
explanation we had been using for it.

  paired delta vs dense (pts), 5 seeds, channel-wise, boost off

    labels    k=0.05    k=0.10    k=0.20    k=0.40
       100     -1.98     -1.46     -0.79     -2.09
       300     -0.75     +1.06     +1.55     -1.03
       600     -0.08     +0.84     +1.43     -0.69
     3,000     -0.80     +0.06     +0.19     -0.05

Two things follow, and both are inconvenient.

1. It is an interior optimum in k, at EVERY budget, and k* = 0.20 does not move
   across a 30x range of label budgets. So sparsity is not a prior whose strength
   scales with data scarcity -- k is an architecture constant.

2. **Pattern separation does not explain the effect.** The sparsest arm has the
   HIGHEST code separation at every budget (k=0.05: 0.227 -> 0.298) and the WORST
   accuracy. Across all 16 kWTA cells the correlation between separation and
   paired delta is r = +0.19 (p = 0.48) -- nothing. Separation is available in
   abundance and buying it harder makes the model worse.

So the "sparse codes -> pattern separation -> data efficiency" story, which
survived v2 and looked confirmed, is dead as a causal account. Something else is
producing the +1.43, and an inverted U in k is the signature of a
benefit/cost tradeoff, not of a representational mechanism.

THE QUESTION THIS MODULE ANSWERS
--------------------------------
Is the *input-dependent competition* doing the work, or would any sparsification
with the same structure do just as well?

`RandomKChannel` is `KWinnersTakeAllV2(dim="channel")` with exactly one thing
removed: instead of keeping the k channels with the largest activations at each
spatial position, it keeps k channels chosen uniformly at random. Same k, same
axis, same number of surviving units, same tensor shapes, same train/eval
behaviour. The ONLY difference is whether selection depends on the input.

  * kWTA >> random-k  -> the competition is the mechanism. Winners are selected
                         for a reason and that reason matters.
  * kWTA ~= random-k  -> the competition is decorative. What helps is the
                         structured sparsity itself, and the honest description
                         of the result is "a well-tuned structured regularizer,"
                         not lateral inhibition. The brain-inspired framing would
                         then be doing no work and should be dropped.

This is the control a reviewer asks for first, and it is cheap. It should have
been run before any of the interpretation in the previous rounds.

IMPLEMENTATION NOTE
-------------------
Selection uses `topk` over a uniform random tensor rather than sampling, which
guarantees *exactly* k winners per position -- matching kWTA's count exactly
rather than in expectation, the way dropout would. Masking is applied in both
train and eval mode, again matching kWTA, so the comparison is not confounded by
one arm being stochastic at inference and the other deterministic. That does make
this arm's test accuracy slightly stochastic; over the 10k-example test set the
resulting variance is far below the seed variance, and the run is reproducible
because the global seed is set before each run.
"""

import torch
import torch.nn as nn


class RandomKChannel(nn.Module):
    """
    Keeps exactly k randomly-chosen channels at each spatial position (4-D input)
    or k randomly-chosen units per sample (2-D input).

    Structurally identical to KWinnersTakeAllV2(dim="channel"); selection is
    random instead of competitive.
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
            # topk over uniform noise == exactly k uniformly-chosen winners
            idx = torch.rand_like(x).topk(k, dim=dim).indices
        mask = torch.zeros_like(x).scatter_(dim, idx, 1.0)
        return x * mask

    def extra_repr(self) -> str:
        return f"k={self.k}, selection=random"


class StructuredDropout(nn.Module):
    """
    The conventional analogue, for the "isn't this just spatial dropout?"
    objection -- which a reviewer will raise and which deserves a real answer
    rather than an argument.

    Drops whole channels (`Dropout2d` semantics) for 4-D input and individual
    units (`Dropout`) for 2-D input, with the standard train-only + rescale
    behaviour. `p` is set so the expected surviving fraction matches kWTA's k.

    Note the deliberate contrast with `RandomKChannel`: dropout is train-only,
    rescaled, and stochastic in its count; random-k is always-on, unrescaled, and
    exact. Between them they bracket "structured sparsity without competition."
    """

    def __init__(self, p: float):
        super().__init__()
        self.p = float(p)
        self.drop2d = nn.Dropout2d(p)
        self.drop1d = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2d(x) if x.dim() == 4 else self.drop1d(x)

    def extra_repr(self) -> str:
        return f"p={self.p}"
