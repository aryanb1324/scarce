"""
Paired statistics and the decision rule.

The rule this module enforces is the one the project learned the hard way, twice:

  * A 5-seed run reported kWTA at -0.79 pts and it was written up as a finding.
    Twenty seeds gave -0.06 +/- 0.31. The first five seeds reproduced -0.794
    exactly -- it was seed luck, and the paired sd at that budget (1.375) meant
    resolving 0.79 pt needed ~24 seeds, not 5.

  * At n = 3, p-values tracked luck in the spread rather than effect size: a
    -0.19 pt effect came out "significant" and a -1.58 pt effect did not.

So the decision rule refuses to name a winner unless the effect clears the noise
floor MEASURED ON THE CALLER'S OWN DATA, and it reports the seeds that would be
needed when it cannot. Returning "use dense" is a real answer, not a failure.

Comparisons are PAIRED: each seed contributes one delta against dense at the same
seed, same data, same batch order. Paired deltas have far lower variance than two
independent means -- reporting the arms separately throws that away.

No scipy: the regularized incomplete beta below gives exact t-distribution
p-values, so the library stays dependency-light (torch + numpy only).
"""

from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Optional, Sequence

# 80% power at alpha=0.05, two-sided: (z_{0.975} + z_{0.80})^2 = (1.96 + 0.84)^2
_POWER_Z = 1.96 + 0.84


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny, eps, itmax = 1e-30, 3e-16, 300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    front = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_pvalue(t: float, df: int) -> float:
    """Two-sided p-value for a t statistic."""
    if df <= 0 or not math.isfinite(t):
        return float("nan")
    return _betai(0.5 * df, 0.5, df / (df + t * t))


class Paired(NamedTuple):
    """One arm's paired comparison against dense."""
    name: str
    n: int
    mean: float          # mean paired delta, percentage points
    sd: float            # the noise floor on THIS data
    se: float
    t: float
    p: float
    signs: str           # per-seed sign string, e.g. "++++-+++"
    sign_fraction: float
    mde: float           # minimum effect this n could detect at 80% power
    deltas: List[float]


def paired_stats(name: str, deltas: Sequence[float]) -> Paired:
    """Mean, sd, se, t, p, sign consistency, and the minimum detectable effect."""
    d = list(deltas)
    n = len(d)
    mean = sum(d) / n if n else float("nan")
    if n < 2:
        return Paired(name, n, mean, float("nan"), float("nan"), float("nan"),
                      float("nan"), "+" if mean > 0 else "-",
                      1.0 if mean > 0 else 0.0, float("nan"), d)
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
    se = sd / math.sqrt(n)
    t = mean / se if se > 0 else float("inf") if mean else 0.0
    p = t_pvalue(t, n - 1)
    signs = "".join("+" if x > 0 else "-" for x in d)
    frac = signs.count("+") / n
    mde = _POWER_Z * sd / math.sqrt(n)
    return Paired(name, n, mean, sd, se, t, p, signs, frac, mde, d)


def seeds_needed(effect: float, sd: float) -> Optional[int]:
    """Seeds required to resolve `effect` at 80% power, alpha 0.05 two-sided."""
    if not (math.isfinite(effect) and math.isfinite(sd)) or effect <= 0:
        return None
    return max(2, math.ceil((_POWER_Z * sd / effect) ** 2))


class Decision(NamedTuple):
    winner: str
    reason: str
    confident: bool
    alpha: float
    tied_with: List[str] = []


def paired_difference(a: Paired, b: Paired) -> Paired:
    """Paired comparison between two arms, seed by seed.

    Both arms' deltas are measured against dense at the same seed, so dense
    cancels and `a.deltas[i] - b.deltas[i]` is exactly the paired difference
    between the two mechanisms. That makes head-to-head comparison free -- no
    extra runs -- and it is what tells you whether a winner's margin over the
    runner-up is real or is noise.
    """
    return paired_stats(
        "{} - {}".format(a.name, b.name),
        [x - y for x, y in zip(a.deltas, b.deltas)])


def decide(arms: Sequence[Paired], alpha: float = 0.05,
           min_sign_fraction: float = 0.75,
           complexity: Optional[Dict[str, int]] = None) -> Decision:
    """Pick a winner, or return dense when nothing clears the bar.

    Three conditions, all required:
      1. the mean paired delta is positive;
      2. p < alpha, Bonferroni-corrected for the number of arms tested -- the
         search tries several mechanisms, and testing k of them against one
         reference inflates the false-positive rate if uncorrected;
      3. at least `min_sign_fraction` of seeds agree in sign, so a single lucky
         seed cannot carry the mean.

    When several arms pass, the one with the highest mean is only declared the
    winner if its margin over the others is itself resolvable. Arms whose paired
    head-to-head difference is not significant are TIED with it, and the tie is
    broken toward the lowest `complexity` -- the simpler, better-understood
    component. Shipping an exotic mechanism because its mean was 0.2 pts higher
    than dropout, on data where 0.2 pts is noise, is precisely the overclaim this
    library exists to prevent, and it would be self-serving here.
    """
    tested = [a for a in arms if a.name != "dense"]
    if not tested:
        return Decision("dense", "No candidate arms were run.", False, alpha)

    if any(a.n < 2 for a in tested):
        return Decision(
            "dense",
            "Only {} seed(s): variance is unmeasurable, so no effect can be "
            "distinguished from seed luck. Use seeds>=5.".format(tested[0].n),
            False, alpha)

    adj = alpha / len(tested)
    passing = [a for a in tested
               if a.mean > 0 and a.p < adj and a.sign_fraction >= min_sign_fraction]

    if not passing:
        best = max(tested, key=lambda a: a.mean)
        need = seeds_needed(abs(best.mean), best.sd) if best.mean > 0 else None
        extra = ""
        if need and need > best.n:
            extra = (" The strongest arm ({}, {:+.2f} pts) would need ~{} seeds "
                     "to resolve at this noise level; you ran {}.".format(
                         best.name, best.mean, need, best.n))
        return Decision(
            "dense",
            "No mechanism beat dense by more than the noise on this data "
            "(Bonferroni alpha={:.4f}).{}".format(adj, extra),
            False, alpha)

    best = max(passing, key=lambda a: a.mean)

    # Everything statistically indistinguishable from the best arm, head to head.
    tied = [a for a in passing
            if a.name != best.name and not (paired_difference(best, a).p < adj)]

    rank = complexity or {}
    pool = [best] + tied
    win = min(pool, key=lambda a: (rank.get(a.name, 5), -a.mean))

    reason = ("{:+.2f} pts vs dense, p={:.4g} < {:.4g} (Bonferroni), {}/{} seeds "
              "agree.".format(win.mean, win.p, adj, win.signs.count("+"), win.n))
    others = [a.name for a in pool if a.name != win.name]
    if others:
        reason += (" Statistically tied with {}; picked the simplest of them, so "
                   "an unresolvable margin does not sell you a fancier component."
                   .format(", ".join(others)))
    return Decision(win.name, reason, True, alpha, others)
