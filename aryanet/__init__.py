"""
aryanet -- pick a model that actually works when you don't have much data.

    import aryanet
    result = aryanet.fit(x_train, y_train, x_test=x_test, y_test=y_test)
    print(result.report())
    preds = result.predict(x_new)

WHAT THIS DOES, AND WHAT IT DOES NOT CLAIM
------------------------------------------
It does NOT ship a magic data-efficient architecture. The honest state of the
evidence: channel-wise kWTA beat a dense baseline by +1.6 points at 300-600 MNIST
labels (8/8 seeds), was worth ~0 at 100 labels and +0.19 at 3,000, and was tied
by ordinary dropout at one of the two budgets where it worked -- all on one
dataset that saturates near 99% and flatters regularizers of every kind.

What it DOES ship is the measurement. `fit` runs each candidate mechanism on your
data under a protocol built to avoid the mistakes that produced wrong answers in
this project's own history -- train to convergence rather than fixed epochs,
paired seeds on identical batches, normalization fitted on training data only --
measures the noise floor on your data, and names a winner only when the effect
clears it. When nothing does, it says so and gives you a dense network.

That refusal is the feature. A library that always finds a winner is a library
that will hand you seed luck.
"""

from aryanet.mechanisms import Candidate, control_candidates, default_candidates
from aryanet.nets import build_net
from aryanet.protocol import Normalizer, TrainConfig, stratified_split
from aryanet.search import SearchResult, fit
from aryanet.stats import Paired, decide, paired_stats, seeds_needed

__version__ = "0.1.0"

__all__ = [
    "fit",
    "SearchResult",
    "TrainConfig",
    "Candidate",
    "default_candidates",
    "control_candidates",
    "build_net",
    "Normalizer",
    "stratified_split",
    "paired_stats",
    "seeds_needed",
    "decide",
    "Paired",
    "__version__",
]
