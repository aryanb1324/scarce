"""
Screen-then-confirm search, and exactly how it avoids selection contamination.

THE PROBLEM
-----------
Adding an architecture axis multiplies the search space. The curated space is 12
arms; the full cross is 17. At the seed counts this project's own history says
are necessary -- 20 seeds were what retracted the 100-label finding -- 12 arms x
20 seeds is 240 full training runs for one answer. Nobody will run that, so the
tool would be ignored, which is a worse outcome than a slightly weaker test.

So: screen every arm cheaply, then confirm only the survivors properly.

THE TRAP
--------
The obvious implementation is fatally wrong. Run all 12 arms at 3 seeds, keep the
best 3, run those 3 for 5 more seeds, then decide using all 8 seeds per surviving
arm. That is the garden of forking paths. The 3 seeds that chose the arms are
part of the evidence that then judges them, so an arm selected *because* those
seeds happened to favour it carries that luck into its final mean. The selection
step is a maximum over 11 noisy estimates, which is upward-biased by construction,
and pooling the screening seeds back in imports that bias directly into the
number the decision rule tests. The p-value is then not a p-value. This project
has already published one finding that was a maximum-over-noise artifact (the
100-label "reversal", retracted at 20 seeds) and the point of this library is not
to build a faster way of doing that.

HOW THIS AVOIDS IT: DISJOINT SEEDS, AND THE SCREENING SEEDS ARE DISCARDED
------------------------------------------------------------------------
1. Stage 1 runs every arm on seeds `0 .. S1-1` and produces a RANKING.
2. Stage 2 runs the survivors on seeds `S1 .. S1+S2-1` -- FRESH seeds, disjoint
   from stage 1, never previously observed for any arm.
3. The decision is computed from stage-2 deltas ONLY. Stage-1 deltas are reported
   in the output, clearly labelled as screening, and are not pooled in. The code
   enforces this structurally: `decide` is handed the stage-2 `Paired` objects and
   the stage-1 ones are never in scope at that call site.

This is sample splitting ("screen and clean"). Under the null that an arm is no
better than the reference, its stage-2 deltas are independent of everything stage
1 observed -- different seeds mean different initializations and different batch
orders -- so conditioning on having been selected does not shift the stage-2
sampling distribution. The type-I error rate of the stage-2 test is therefore the
nominal one, and Bonferroni over the arms actually tested in stage 2 (3-4) is
valid rather than over all 12 screened. That is not a loophole, it is the payoff:
the screen buys a LOOSER correction on the confirming test at the price of some
power to notice an arm it wrongly dropped.

THIS IS MEASURED, NOT ARGUED
----------------------------
The paragraph above is a claim about type-I error, and this project's rule is
that a claim gets a measurement. `tests/test_architectures.py::
test_the_staged_procedure_controls_false_positives_and_pooling_does_not`
simulates a world where every arm is genuinely identical to the reference, draws
paired deltas at the sd measured on real 300-label MNIST runs (0.87), and counts
how often each procedure declares a winner. At 4,000 replicates of the full
12-arm search:

    staged, disjoint seeds (as implemented)   0.026    <- at or below nominal
    staged, POOLED seeds (the trap)           0.060    <- above nominal 0.05
    no screening, all 12 arms at 8 seeds      0.025    <- the reference point

So the implementation costs nothing in validity against running everything, and
the obvious wrong version inflates the false-positive rate by 2.3x. The test
fails if the seed split ever stops working.

Two supporting details, both load-bearing:

  * THE REFERENCE ARM IS NEVER SCREENED OUT. It is carried into stage 2
    unconditionally, regardless of how it ranked. Selecting the reference on
    stage-1 performance would bias every stage-2 delta downward, since deltas are
    measured against it -- the contamination would re-enter through the baseline
    rather than through the treatment.

  * THE SCREEN RANKS, IT DOES NOT TEST. It sorts by mean stage-1 delta and keeps
    the top `keep`. It applies no significance threshold and no sign rule, and it
    does not drop arms for having a negative mean. At 2-3 seeds a p-value tracks
    luck in the spread rather than effect size -- measured in this repo: a -0.19
    pt effect came out "significant" at n=3 and a -1.58 pt effect did not. A
    screen that tested would be a random number generator with a threshold.

WHAT THIS DOES NOT FIX -- STATED PLAINLY
----------------------------------------
Seed splitting removes the selection bias that comes from TRAINING randomness. It
does not remove selection bias that comes from the DATA, because both stages use
the same train/validation split. If an arm is favoured by a quirk of this
particular validation set, it will be favoured again in stage 2 and the test will
not catch it. So the guarantee is: valid inference about which arm is better ON
THIS DATASET AND THIS SPLIT, not about which arm is better on the population the
dataset was drawn from. That is the same conditional claim the single-stage search
makes -- `fit`'s whole premise is "measured on YOUR data" -- so staging does not
weaken it, but it does not strengthen it either, and no amount of seed splitting
will. The fix for that is a held-out test set, which is why `fit` asks for one and
reports it separately.

Second admitted limit: the returned `.model` is the best-validation checkpoint
across all runs, chosen on the same validation set the decision used. Its
validation accuracy is optimistically biased, in both single-stage and staged
mode. `test_acc` is the only unbiased number in the report.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Sequence, Union

from aryanet.stats import Paired

#: Arm-name column width. `cnn_narrow/kwta_channel_k0.2` is 28 characters, and a
#: table whose columns shift when a name overflows is a table people misread.
NAME_W = 30


class Budget(NamedTuple):
    """A compute setting: how wide the screen, how deep the confirmation.

    `runs(n_arms)` is the exact number of full training runs, which is the only
    honest unit of cost here -- runs differ in wall-clock by architecture (a
    linear arm is seconds, a wide CNN with kWTA is minutes) so quoting minutes
    would be a fiction that depends on the search space.
    """
    name: str
    screen_seeds: int
    keep: int
    confirm_seeds: int

    def survivors(self, n_arms: int) -> int:
        return min(self.keep, max(0, n_arms - 1)) + 1        # + the reference

    def runs(self, n_arms: int) -> int:
        return n_arms * self.screen_seeds + self.survivors(n_arms) * self.confirm_seeds

    def pays_off(self, n_arms: int) -> bool:
        """Whether staging is cheaper here than running every arm at full seeds.

        It is not always. A space small enough that the screen eliminates almost
        nothing pays for the screening runs and gets no saving: 2 arms under
        `quick` costs 14 runs against 10 for the direct search. Worth saying out
        loud -- charging someone extra compute for a weaker test would be the
        tool taking their time to look sophisticated.
        """
        return self.runs(n_arms) < n_arms * self.confirm_seeds

    def describe(self, n_arms: int) -> str:
        naive = n_arms * self.confirm_seeds
        text = ("budget={}: {} arms x {} screening seeds + {} survivors x {} "
                "confirming seeds = {} runs (vs {} to run every arm at {} seeds)"
                .format(self.name, n_arms, self.screen_seeds,
                        self.survivors(n_arms), self.confirm_seeds,
                        self.runs(n_arms), naive, self.confirm_seeds))
        if not self.pays_off(n_arms):
            text += ("\n  NOTE: staging costs MORE than the direct search on a "
                     "space this small and buys\n  a weaker test. Use "
                     "budget=None, seeds={} instead.".format(self.confirm_seeds))
        return text


#: The three presets. Cost in full training runs, for the curated 12-arm space:
#:
#:   quick     12x2 + 3x5  =  39 runs   (naive equivalent: 60)   direction check
#:   standard  12x3 + 4x8  =  68 runs   (naive: 96)              a conclusion
#:   thorough  12x5 + 5x20 = 160 runs   (naive: 240)             a claim
#:
#: `confirm_seeds` is what the decision actually rests on, so read the budget as
#: "a 5 / 8 / 20-seed conclusion". 5 is the floor at which the rule will consider
#: naming a winner at all; 20 is what it took to retract the 100-label finding,
#: and is the right setting at a noisy budget where a decision hinges on it.
BUDGETS = {
    "quick": Budget("quick", screen_seeds=2, keep=2, confirm_seeds=5),
    "standard": Budget("standard", screen_seeds=3, keep=3, confirm_seeds=8),
    "thorough": Budget("thorough", screen_seeds=5, keep=4, confirm_seeds=20),
}


def resolve_budget(budget: Union[str, Budget]) -> Budget:
    if isinstance(budget, Budget):
        return budget
    try:
        return BUDGETS[str(budget).lower()]
    except KeyError:
        raise ValueError(
            "budget must be one of {} or a Budget instance; got {!r}".format(
                sorted(BUDGETS), budget))


class Stage(NamedTuple):
    """One stage's record: which seeds it used and what it saw."""
    name: str
    seeds: List[int]
    stats: List[Paired]


def screen(stage1: Sequence[Paired], keep: int, reference: str,
           complexity: Optional[Dict[str, int]] = None) -> List[str]:
    """Names to carry into stage 2: the top `keep` by mean, plus the reference.

    Ranking only. No p-value, no sign rule, no sign filter -- see the module
    docstring for why a significance test on 2-3 seeds would be noise with a
    threshold on it. The reference is unconditional; screening it on performance
    would bias every stage-2 delta through the baseline.

    TIES BREAK TOWARD LOWER COMPLEXITY, matching the decision rule. Without this
    the screen breaks ties alphabetically, which silently eliminates the simpler
    arm before it ever gets a fair test -- and on saturated data, where many arms
    tie exactly at the screen's low seed count, that happens constantly. A
    library whose cheap stage quietly discards the linear baseline and whose
    expensive stage would have preferred it is not measuring, it is filtering
    toward its own mechanisms.
    """
    rank = complexity or {}
    tested = [a for a in stage1 if a.name != reference]
    ranked = sorted(tested,
                    key=lambda a: (-a.mean, rank.get(a.name, 10 ** 6), a.name))
    return [reference] + [a.name for a in ranked[:max(0, keep)]]


def dropped(stage1: Sequence[Paired], kept: Sequence[str]) -> List[Paired]:
    """Arms the screen eliminated, worst first -- reported, never silently gone."""
    keep = set(kept)
    return sorted([a for a in stage1 if a.name not in keep], key=lambda a: a.mean)


def contamination_check(stage1: Sequence[Paired],
                        stage2: Sequence[Paired],
                        seeds1: Sequence[int],
                        seeds2: Sequence[int]) -> Optional[str]:
    """Return a description of any contamination found, or None if clean.

    Cheap enough to run on every staged search, and it is run on every staged
    search. A structural guarantee that is never checked is a comment.

    Fails if the two stages share a seed, or if any arm's stage-2 delta list has
    absorbed stage-1 observations (detected by length: stage 2 must contribute
    exactly `len(seeds2)` deltas per arm, no more).
    """
    overlap = sorted(set(seeds1) & set(seeds2))
    if overlap:
        return ("stage 1 and stage 2 share seed(s) {}: the runs that selected "
                "the arms would also be judging them".format(overlap))
    bad = [a.name for a in stage2 if a.n != len(seeds2)]
    if bad:
        return ("stage-2 statistics for {} were computed from {} deltas but "
                "stage 2 ran {} seeds -- screening seeds appear to have been "
                "pooled in".format(bad, [a.n for a in stage2 if a.name in bad],
                                   len(seeds2)))
    return None


def format_stage(stage: Stage, reference: str, title: str,
                 note: str = "") -> List[str]:
    """Render one stage's table. Screening tables print NO p-values.

    Deliberate: a p-value on 2-3 seeds is the number this project has twice been
    misled by, and printing it next to a mean is an invitation to read it. The
    screening table shows the mean and the sign string, which is what a ranking
    is entitled to.
    """
    lines = [title, "  seeds {}".format(_seed_range(stage.seeds))]
    if note:
        lines.append("  {}".format(note))
    show_p = stage.name != "screen"
    lines.append("  {:<{w}}{:>9}{:>7}{:>8}  {}".format(
        "arm", "delta", "sd", "p" if show_p else "", "signs", w=NAME_W))
    for a in stage.stats:
        if a.name == reference:
            lines.append("  {:<{w}}{}".format(
                a.name, "(reference -- every delta is measured against this)",
                w=NAME_W))
        elif show_p:
            lines.append("  {:<{w}}{:>+9.2f}{:>7.2f}{:>8.4f}  {}".format(
                a.name, a.mean, a.sd, a.p, a.signs, w=NAME_W))
        else:
            lines.append("  {:<{w}}{:>+9.2f}{:>7.2f}{:>8}  {}".format(
                a.name, a.mean, a.sd, "", a.signs, w=NAME_W))
    return lines


def _seed_range(seeds: Sequence[int]) -> str:
    s = list(seeds)
    if not s:
        return "(none)"
    return "{}-{}".format(s[0], s[-1]) if len(s) > 1 else str(s[0])
