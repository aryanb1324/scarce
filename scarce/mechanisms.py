"""
The candidate mechanisms the search chooses between.

Each candidate is an *activation factory* -- a zero-arg callable returning the
module that replaces every ReLU. That is the whole interface, which is what makes
adding a mechanism cheap and makes every arm architecturally identical.

WHAT THE EVIDENCE ACTUALLY SAYS, per candidate. These strings are printed in the
report, because a user choosing a mechanism deserves to know how well-supported
it is rather than inferring confidence from the fact that it shipped.

The measurements behind them are MNIST-only, at 300-600 labels, 8 seeds, paired
(experiments/results/20260728_151333_stage2_control). They are the reason this
library searches instead of asserting: `kwta_channel` beat dense by +1.6 pts
there, and `dropout` tied it at one of those two budgets. Neither is a safe
default for data nobody has measured -- yours.
"""

from __future__ import annotations

from typing import Callable, List, NamedTuple

import torch.nn as nn

from architecture.modules.kwta_v2 import KWinnersTakeAllV2
from architecture.modules.randk import RandomKChannel, StructuredDropout


class Candidate(NamedTuple):
    """One arm of the search.

    `complexity` breaks statistical ties toward the simpler, better-understood
    component. When two arms are indistinguishable on the caller's data, shipping
    the exotic one because its mean was 0.2 pts higher would be exactly the
    overclaim this library exists to avoid. Lower is simpler.
    """
    name: str
    factory: Callable[[], nn.Module]
    evidence: str
    complexity: int = 5


def _kwta(k: float) -> Callable[[], nn.Module]:
    return lambda: nn.Sequential(nn.ReLU(), KWinnersTakeAllV2(k=k, dim="channel",
                                                              boost=0.0))


def _dropout(k: float) -> Callable[[], nn.Module]:
    return lambda: nn.Sequential(nn.ReLU(), StructuredDropout(p=1.0 - k))


def _randk(k: float) -> Callable[[], nn.Module]:
    return lambda: nn.Sequential(nn.ReLU(), RandomKChannel(k=k))


def default_candidates() -> List[Candidate]:
    """The arms searched unless the caller overrides them.

    `dense` is always first and is the reference every delta is measured against.
    Two k values for kWTA: k* = 0.20 did not move across a 30x label range on
    MNIST, but that is one dataset, and the claim "k is an architecture constant"
    has only ever been tested there.
    """
    return [
        Candidate(
            "dense", lambda: nn.ReLU(),
            "Standard ReLU. The reference arm and the fallback when nothing "
            "clears the noise floor.", 0),
        Candidate(
            "kwta_channel_k0.2", _kwta(0.2),
            "Channel-wise k-winner-take-all, k=0.20. +1.58/+1.66 pts vs dense at "
            "300/600 MNIST labels, 8/8 seeds. ~0 at 100 labels and +0.19 at "
            "3,000 -- a narrow window. Not novel: `local=True` in nupic.torch.", 2),
        Candidate(
            "kwta_channel_k0.1", _kwta(0.1),
            "Same, k=0.10. Weaker than k=0.20 on MNIST (+1.06/+0.84) but k* has "
            "only been swept on one dataset.", 2),
        Candidate(
            "dropout_k0.2", _dropout(0.2),
            "Conventional Dropout2d/Dropout at matched surviving fraction. TIED "
            "kwta at 300 MNIST labels (-0.11, n.s.) and lost at 600 (+1.01). A "
            "first-class arm, not a straw man.", 1),
    ]


def control_candidates() -> List[Candidate]:
    """Diagnostic arms, off by default -- they answer 'why', not 'which'.

    `randk` is the control for "is the input-dependent competition doing the work,
    or would any structured sparsity do?" Read `architecture/skills.md` before
    interpreting it: it matches kWTA on nominal count but NOT on surviving signal
    (26.0 vs 7.6 live units, 3.4x), because random selection lands on post-ReLU
    zeros and top-k cannot. Any kwta-minus-randk gap is an upper bound on the
    competition effect, not a measurement of it.
    """
    return [Candidate(
        "randk_channel_k0.2", _randk(0.2),
        "CONTROL, confounded: not sparsity-matched to kwta (3.4x fewer live "
        "units). Upper bound on the competition effect, not a measurement.", 3)]
