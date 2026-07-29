"""
`aryanet.fit` -- measure what actually helps on YOUR data, then use that.

The library does not assert that sparse activations help you. On the one dataset
where this has been measured properly, channel-wise kWTA beat a dense baseline by
+1.6 points at 300-600 labels, was worth nothing at 100 or 3,000, and was TIED by
ordinary dropout at one of the two budgets where it worked. Shipping that as a
default would be an overclaim.

So the selection is the product: run every candidate under a protocol that cannot
lie to you (train to convergence, paired seeds, identical batches), measure the
noise floor on your data, and name a winner only if it clears that floor. "Use
dense" is a frequent and correct answer.

TWO SEARCH AXES
---------------
  MECHANISM  -- the activation: dense, kWTA, dropout. One architecture, so a
                difference cannot be a model-size difference.
  ARCHITECTURE -- capacity and depth, plus a linear model. Opt in with
                `architectures="default"`. At a few hundred labels this is
                usually the DOMINANT knob, and "train a smaller network" or "use
                logistic regression" are answers the mechanism axis alone cannot
                reach. See `aryanet/architectures.py`.

Crossing the axes multiplies runs, so `budget=` switches on a two-stage
screen-then-confirm search: rank every arm on cheap seeds, then confirm the
survivors on FRESH, DISJOINT seeds and decide on those alone. The contamination
argument, and what seed-splitting does not fix, is in `aryanet/staged.py`.
"""

from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Union

import torch
import torch.nn as nn

from aryanet.architectures import Arm, Architecture, resolve_arms
from aryanet.mechanisms import Candidate, control_candidates
from aryanet.nets import count_parameters
from aryanet.protocol import (
    Normalizer,
    TrainConfig,
    as_float_tensor,
    evaluate,
    set_seed,
    stratified_split,
    train_to_convergence,
)
from aryanet.staged import (
    NAME_W,
    Budget,
    Stage,
    contamination_check,
    dropped,
    format_stage,
    resolve_budget,
    screen,
)
from aryanet.stats import Decision, Paired, decide, paired_stats, seeds_needed


class SearchResult(NamedTuple):
    """What the search found, and how much to trust it."""
    winner: str
    decision: Decision
    arms: List[Paired]
    model: nn.Module
    normalizer: Normalizer
    per_arm_acc: Dict[str, float]
    cost: Dict[str, Dict[str, float]]
    test_acc: Optional[float]
    n_train: int
    n_val: int
    seeds: int
    reference: str = "dense"
    stages: List[Stage] = []
    eliminated: List[Paired] = []
    budget: Optional[Budget] = None
    runs: int = 0

    def predict(self, x: Any) -> torch.Tensor:
        """Class predictions for new data, with the fitted normalization applied."""
        xt = self.normalizer(as_float_tensor(x))
        self.model.eval()
        with torch.no_grad():
            return self.model(xt).argmax(dim=1)

    def report(self) -> str:
        return _format_report(self)

    def __repr__(self) -> str:
        return "SearchResult(winner={!r}, confident={}, n_train={})".format(
            self.winner, self.decision.confident, self.n_train)


def fit(x_train: Any, y_train: Any,
        x_val: Any = None, y_val: Any = None,
        x_test: Any = None, y_test: Any = None,
        seeds: int = 5,
        candidates: Optional[Sequence[Candidate]] = None,
        architectures: Union[None, str, Sequence[Architecture]] = None,
        budget: Union[None, str, Budget] = None,
        include_controls: bool = False,
        config: TrainConfig = TrainConfig(),
        val_per_class: int = 50,
        device: Optional[torch.device] = None,
        verbose: bool = True) -> SearchResult:
    """Search architectures and mechanisms on your data; return what actually won.

    Args:
      x_train, y_train: features and integer labels. Images as (N, C, H, W),
        tabular as (N, F). numpy or torch, uint8 or float.
      x_val, y_val: optional. If omitted, a stratified split is held out of
        training data (`val_per_class` per class) -- and note that shrinks the
        training set, which matters when labels are the scarce resource.
      x_test, y_test: optional but recommended. The winner is CHOSEN on
        validation, so its validation number is optimistically biased; a test set
        is the only way to get an honest final figure.
      seeds: paired repetitions, when running single-stage. Below 5 the variance
        estimate is too unstable to support a conclusion and the decision rule
        will usually decline to call one. Ignored when `budget` is set, which
        specifies its own seed counts.
      candidates: override the mechanism arms. Must include `dense`.
      architectures: None (default) searches mechanisms only on the frozen
        research stack -- the original behaviour, unchanged. `"default"` adds the
        capacity axis as a curated 12-arm space; `"full"` is the complete cross
        product; or pass your own `Architecture` sequence.
      budget: `"quick"`, `"standard"`, `"thorough"`, or a `Budget`. Switches on
        the two-stage screen-then-confirm search, whose confirming stage uses
        seeds disjoint from the screen so selection cannot contaminate the test
        (`aryanet/staged.py`). Costs, on the 12-arm space:
            quick     39 runs, a 5-seed conclusion  (naive: 60)
            standard  68 runs, an 8-seed conclusion (naive: 96)
            thorough  160 runs, a 20-seed claim     (naive: 240)
        Leave it None to run every arm at `seeds`, which is fine for a small
        space and gets expensive fast with the architecture axis on.
      include_controls: add diagnostic arms (randk) that answer "why", not
        "which". Read architecture/skills.md before interpreting them.

    Returns a SearchResult whose `.model` is trained and ready, and whose
    `.report()` states the effect size, the noise floor, the parameter cost, and
    what it could not resolve.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xtr = as_float_tensor(x_train)
    ytr = torch.as_tensor(y_train).long().reshape(-1)
    if xtr.shape[0] != ytr.shape[0]:
        raise ValueError("x_train has {} rows but y_train has {}".format(
            xtr.shape[0], ytr.shape[0]))

    if x_val is None:
        tr_idx, va_idx = stratified_split(ytr, val_per_class)
        xva, yva = xtr[va_idx], ytr[va_idx]
        xtr, ytr = xtr[tr_idx], ytr[tr_idx]
    else:
        xva = as_float_tensor(x_val)
        yva = torch.as_tensor(y_val).long().reshape(-1)

    # Fitted on training data only -- statistics from held-out data leak.
    norm = Normalizer.fit(xtr)
    xtr_n, xva_n = norm(xtr), norm(xva)
    xte_n = yte = None
    if x_test is not None:
        xte_n = norm(as_float_tensor(x_test))
        yte = torch.as_tensor(y_test).long().reshape(-1)

    input_shape = tuple(xtr.shape[1:])
    num_classes = int(ytr.max().item()) + 1

    controls = control_candidates() if include_controls else ()
    arms = resolve_arms(candidates, architectures, controls)
    ref = next((a.name for a in arms if a.is_reference), None)
    if ref is None:
        raise ValueError(
            "the search space must include a 'dense' reference arm -- every "
            "delta is measured against it.")

    rank = {a.name: a.complexity for a in arms}
    bud = resolve_budget(budget) if budget is not None else None

    if verbose:
        print("aryanet: {} train / {} val, shape {}, {} classes".format(
            xtr.shape[0], xva.shape[0], input_shape, num_classes))
        if bud is None:
            print("         {} arms x {} seeds = {} runs, device={}".format(
                len(arms), seeds, len(arms) * seeds, device))
        else:
            print("         {}".format(bud.describe(len(arms))))
            print("         device={}".format(device))
        print()

    state = _State(input_shape, num_classes, xtr_n, ytr, xva_n, yva,
                   config, device, verbose)

    stages: List[Stage] = []
    eliminated: List[Paired] = []

    if bud is None:
        seed_list = list(range(seeds))
        _run_stage(state, arms, seed_list, ref)
        final = _stats(state, arms, seed_list, ref)
        n_seeds, runs = seeds, len(arms) * seeds
    else:
        s1 = list(range(bud.screen_seeds))
        s2 = list(range(bud.screen_seeds,
                        bud.screen_seeds + bud.confirm_seeds))

        if verbose:
            print("STAGE 1 -- screening {} arms on seeds {}-{} (ranking only)\n"
                  .format(len(arms), s1[0], s1[-1]))
        _run_stage(state, arms, s1, ref)
        screen_stats = _stats(state, arms, s1, ref)
        keep = screen(screen_stats, bud.keep, ref, complexity=rank)
        survivors = [a for a in arms if a.name in set(keep)]
        eliminated = dropped(screen_stats, keep)
        stages.append(Stage("screen", s1, screen_stats))

        if verbose:
            print("\n  screen keeps: {}".format(", ".join(keep)))
            print("  screen drops: {}\n".format(
                ", ".join(a.name for a in eliminated) or "(none)"))
            print("STAGE 2 -- confirming {} arms on FRESH seeds {}-{}\n"
                  "           (the decision uses these seeds only)\n".format(
                      len(survivors), s2[0], s2[-1]))
        _run_stage(state, survivors, s2, ref)
        # `final` is built from stage-2 seeds only. Stage-1 deltas are reported
        # but never pooled: pooling them would import the selection bias that
        # splitting the seeds exists to remove.
        final = _stats(state, survivors, s2, ref)
        stages.append(Stage("confirm", s2, final))

        problem = contamination_check(screen_stats, final, s1, s2)
        if problem is not None:                       # pragma: no cover - guard
            raise RuntimeError(
                "staged search is contaminated: {}. Refusing to report a "
                "p-value that does not mean what it says.".format(problem))

        n_seeds, runs = bud.confirm_seeds, bud.runs(len(arms))
        seed_list = s2

    decision = decide(final, complexity=rank, reference=ref)

    # The returned weights are the best-validation checkpoint across EVERY run of
    # the winning arm, screening seeds included. That is deliberate and is not a
    # contamination of the decision -- the decision was already made, from stage-2
    # deltas only, and having chosen an arm you want its best trained instance.
    # It does mean the winner's validation accuracy is optimistically biased,
    # which is true in single-stage mode too and is why `test_acc` exists.
    winner_arm = next(a for a in arms if a.name == decision.winner)
    model = winner_arm.build(input_shape, num_classes)
    model.load_state_dict(state.best_state[decision.winner])
    model.to(device).eval()

    test_acc = None
    if xte_n is not None:
        test_acc = evaluate(model, xte_n, yte, device)

    result = SearchResult(
        winner=decision.winner, decision=decision, arms=final, model=model,
        normalizer=norm,
        # Accuracies come from the SAME seeds the decision used, so the val-acc
        # column and the delta column are the same measurement. Averaging an
        # arm over stage-1 and stage-2 seeds while its delta covers stage 2 only
        # would print two numbers that do not correspond.
        per_arm_acc=_mean_acc(state, seed_list),
        cost=state.costs(arms, input_shape, num_classes),
        test_acc=test_acc, n_train=int(xtr.shape[0]), n_val=int(xva.shape[0]),
        seeds=n_seeds, reference=ref, stages=stages, eliminated=eliminated,
        budget=bud, runs=runs)

    if verbose:
        print(result.report())
    return result


# --------------------------------------------------------------------------
# Running arms. One code path whether the search is single- or two-stage, so a
# staged run cannot accidentally train under different conditions than a plain
# one -- the ONLY difference between stages is which seeds they use.
# --------------------------------------------------------------------------

class _State(object):
    """Accumulated per-(arm, seed) results across every stage."""

    def __init__(self, input_shape, num_classes, xtr, ytr, xva, yva,
                 config, device, verbose):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.xtr, self.ytr, self.xva, self.yva = xtr, ytr, xva, yva
        self.config, self.device, self.verbose = config, device, verbose
        self.acc: Dict[str, Dict[int, float]] = {}
        self.runs: Dict[str, List[Any]] = {}
        self.best_state: Dict[str, Any] = {}
        self.best_val: Dict[str, float] = {}

    def costs(self, arms: Sequence[Arm], input_shape, num_classes
              ) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for arm in arms:
            rs = self.runs.get(arm.name)
            if not rs:
                continue
            out[arm.name] = {
                "params": float(count_parameters(
                    arm.build(input_shape, num_classes))),
                "active_units": sum(r.active_units for r in rs) / len(rs),
                "dead_fraction": sum(r.dead_frac for r in rs) / len(rs),
                "seconds": sum(r.seconds for r in rs) / len(rs),
                "best_step": sum(r.best_step for r in rs) / len(rs),
            }
        return out


def _run_stage(state: _State, arms: Sequence[Arm], seed_list: Sequence[int],
               reference: str) -> None:
    for seed in seed_list:
        for arm in arms:
            set_seed(seed)                     # identical init RNG across arms
            model = arm.build(state.input_shape, state.num_classes)
            r = train_to_convergence(model, state.xtr, state.ytr,
                                     state.xva, state.yva, seed,
                                     state.config, state.device)
            state.acc.setdefault(arm.name, {})[seed] = r.val_acc
            state.runs.setdefault(arm.name, []).append(r)
            if r.val_acc > state.best_val.get(arm.name, -1.0):
                state.best_val[arm.name] = r.val_acc
                state.best_state[arm.name] = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()}
            if state.verbose:
                base = state.acc.get(reference, {}).get(seed)
                delta = "" if base is None or arm.name == reference else \
                    "  ({:+5.2f})".format((r.val_acc - base) * 100)
                print("  seed {} {:<24} val={:.4f}{}  act={:5.1f}  {:4.1f}s"
                      .format(seed, arm.name, r.val_acc, delta,
                              r.active_units, r.seconds))
        if state.verbose:
            print()


def _mean_acc(state: _State, seed_list: Sequence[int]) -> Dict[str, float]:
    """Mean validation accuracy per arm over `seed_list`, where it has runs."""
    out = {}
    for name, by_seed in state.acc.items():
        vals = [by_seed[s] for s in seed_list if s in by_seed]
        if vals:
            out[name] = sum(vals) / len(vals)
    return out


def _stats(state: _State, arms: Sequence[Arm], seed_list: Sequence[int],
           reference: str) -> List[Paired]:
    """Paired deltas against the reference, using ONLY `seed_list`.

    Every arm's delta at a seed is measured against the reference at the SAME
    seed -- same initialization RNG, same batch order, same data. Restricting to
    `seed_list` is what keeps a staged search's confirming statistics free of the
    seeds that did the selecting.
    """
    return [paired_stats(
        a.name,
        [(state.acc[a.name][s] - state.acc[reference][s]) * 100
         for s in seed_list])
        for a in arms]


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _format_report(r: SearchResult) -> str:
    lines = []
    add = lines.append
    add("=" * 78)
    add("ARYANET SEARCH -- paired delta vs {}, percentage points".format(
        r.reference))
    add("=" * 78)
    add("{} training / {} validation examples".format(r.n_train, r.n_val))
    if r.budget is not None:
        add(r.budget.describe(len(r.arms) + len(r.eliminated)))
        add("the decision rests on the {} confirming seeds, not on all {} runs"
            .format(r.seeds, r.runs))
    else:
        add("{} arms x {} seeds = {} runs".format(
            len(r.arms), r.seeds, r.runs))
    add("")

    if r.eliminated:
        stage1 = next(s for s in r.stages if s.name == "screen")
        lines.extend(format_stage(
            stage1, r.reference,
            "STAGE 1 SCREEN -- ranking only, not inference",
            "these numbers chose which arms to confirm; they do NOT enter "
            "the decision"))
        add("")
        add("  eliminated: {}".format(
            ", ".join("{} ({:+.2f})".format(a.name, a.mean)
                      for a in r.eliminated)))
        add("")
        title = "STAGE 2 CONFIRM -- fresh seeds, disjoint from the screen"
        note = "the decision below is made from these deltas alone"
    else:
        title = "RESULTS -- every arm, every seed"
        note = ""

    add(title)
    if note:
        add("  {}".format(note))
    add("  {:<{w}}{:>9}{:>9}{:>7}{:>8}  {}".format(
        "arm", "val acc", "delta", "sd", "p", "signs", w=NAME_W))
    for a in r.arms:
        acc = r.per_arm_acc.get(a.name, float("nan"))
        if a.name == r.reference:
            add("  {:<{w}}{:>9.4f}  (reference)".format(a.name, acc, w=NAME_W))
        else:
            add("  {:<{w}}{:>9.4f}{:>+9.2f}{:>7.2f}{:>8.4f}  {}".format(
                a.name, acc, a.mean, a.sd, a.p, a.signs, w=NAME_W))

    add("")
    add("COST (what each arm spends, not just what it buys)")
    add("  {:<{w}}{:>12}{:>14}{:>12}{:>10}".format(
        "arm", "params", "active units", "dead frac", "sec/run", w=NAME_W))
    for name in sorted(r.cost, key=lambda n: (n != r.reference, n)):
        c = r.cost[name]
        add("  {:<{w}}{:>12,}{:>14.1f}{:>12.3f}{:>10.1f}".format(
            name, int(c["params"]), c["active_units"], c["dead_fraction"],
            c["seconds"], w=NAME_W))
    add("  Parameter count is listed because an architecture search deliberately")
    add("  varies model size: a delta here can be a capacity effect, not a")
    add("  mechanism effect, and you should be able to see which.")

    add("")
    add("-" * 78)
    if r.decision.confident:
        add("WINNER: {}".format(r.winner))
        add("  {}".format(r.decision.reason))
        if r.decision.tied_with:
            add("  Tied arms are genuinely interchangeable here -- if you prefer")
            add("  one for reasons the accuracy cannot see (inference cost, a")
            add("  standard component your team already knows), take it.")
    else:
        add("WINNER: {} (nothing cleared the noise floor)".format(r.reference))
        add("  {}".format(r.decision.reason))
        add("  This is a real answer. On this data, at this label budget, the")
        add("  candidates tested do not beat a standard network.")

    noise = [a for a in r.arms if a.name != r.reference]
    if noise:
        floor = sum(a.sd for a in noise) / len(noise)
        mde = sum(a.mde for a in noise) / len(noise)
        add("")
        add("NOISE FLOOR on this data: paired sd ~{:.2f} pts across {} seeds,".format(
            floor, r.seeds))
        add("  so {} seeds can resolve effects down to ~{:.2f} pts at 80% power.".format(
            r.seeds, mde))
        need = seeds_needed(1.0, floor)
        if need:
            add("  Resolving a 1.0 pt effect here would need ~{} seeds.".format(need))
        if r.budget is not None and need and need > r.seeds:
            add("  budget='thorough' raises the confirming stage to 20 seeds.")

    add("")
    if r.test_acc is not None:
        add("TEST ACCURACY (unbiased): {:.4f}".format(r.test_acc))
    else:
        add("NOTE: the winner was chosen on validation, so its validation number")
        add("  is optimistically biased. Pass x_test/y_test for an honest figure.")
    if r.n_val < 500:
        add("WARNING: only {} validation examples -- accuracy is quantized to".format(
            r.n_val))
        add("  {:.2f} pts per example, which may exceed the effects you care about.".format(
            100.0 / r.n_val))
    add("-" * 78)
    return "\n".join(lines)
