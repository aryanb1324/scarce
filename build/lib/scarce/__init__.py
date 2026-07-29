"""
scarce -- pick a model that actually works when you don't have much data.

    import scarce
    result = scarce.fit(x_train, y_train, x_test=x_test, y_test=y_test)
    print(result.report())
    preds = result.predict(x_new)

WHAT THIS DOES, AND WHAT IT DOES NOT CLAIM
------------------------------------------
It does NOT ship a magic data-efficient architecture. The honest state of the
evidence: channel-wise kWTA beat a dense baseline by +1.6 points at 300-600 MNIST
labels (8/8 seeds), was worth ~0 at 100 labels and +0.19 at 3,000, and was tied
by ordinary dropout at one of the two budgets where it worked -- all on one
dataset that saturates near 99% and flatters regularizers of every kind.

What it DOES ship is the measurement. `fit` runs each candidate on your data under
a protocol built to avoid the mistakes that produced wrong answers in this
project's own history -- train to convergence rather than fixed epochs, paired
seeds on identical batches, normalization fitted on training data only --
measures the noise floor on your data, and names a winner only when the effect
clears it. When nothing does, it says so and gives you a standard network.

That refusal is the feature. A library that always finds a winner is a library
that will hand you seed luck.

TWO AXES, AND A COMPUTE BUDGET
------------------------------
    # mechanisms only, on the frozen research stack (the original behaviour)
    scarce.fit(x, y)

    # + capacity, depth, and a logistic-regression baseline: 12 arms, and a
    #   two-stage search so that costs 68 runs rather than 96
    scarce.fit(x, y, architectures="default", budget="standard")

At a few hundred labels, capacity is usually the dominant knob and a SMALLER
model often wins outright, so a search that varies only the activation cannot
reach the right answer. `budget` switches on a screen-then-confirm search whose
confirming stage uses seeds DISJOINT from the screen, and decides on those alone
-- selecting arms and then testing them on the same runs is a garden-of-forking-
paths error, and `scarce/staged.py` documents both how that is avoided and the
one thing seed-splitting cannot fix.
"""

from scarce.architectures import (
    Architecture,
    Arm,
    cross,
    default_architectures,
    default_arms,
)
from scarce.mechanisms import Candidate, control_candidates, default_candidates
from scarce.nets import build_cnn, build_linear, build_net, count_parameters
from scarce.protocol import Normalizer, TrainConfig, stratified_split
from scarce.search import SearchResult, fit
from scarce.staged import BUDGETS, Budget, screen
from scarce.stats import Paired, decide, paired_stats, seeds_needed

# Single source of truth is pyproject's [project] version, read from installed
# metadata. The fallback covers running from a source checkout that was never
# `pip install`ed; keep it in sync with pyproject only as a last resort.
try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version
    try:
        __version__ = _pkg_version("scarce")
    except PackageNotFoundError:
        __version__ = "0.2.0"
except ImportError:  # pragma: no cover - importlib.metadata exists on 3.9
    __version__ = "0.2.0"

__all__ = [
    "fit",
    "SearchResult",
    "TrainConfig",
    "Budget",
    "BUDGETS",
    "screen",
    "Candidate",
    "default_candidates",
    "control_candidates",
    "Architecture",
    "Arm",
    "default_architectures",
    "default_arms",
    "cross",
    "build_net",
    "build_cnn",
    "build_linear",
    "count_parameters",
    "Normalizer",
    "stratified_split",
    "paired_stats",
    "seeds_needed",
    "decide",
    "Paired",
    "__version__",
]
