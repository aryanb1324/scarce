"""
`aryanet.fit` -- measure which mechanism helps on YOUR data, then use that one.

The library does not assert that sparse activations help you. On the one dataset
where this has been measured properly, channel-wise kWTA beat a dense baseline by
+1.6 points at 300-600 labels, was worth nothing at 100 or 3,000, and was TIED by
ordinary dropout at one of the two budgets where it worked. Shipping that as a
default would be an overclaim.

So the mechanism selection is the product: run every candidate under a protocol
that cannot lie to you (train to convergence, paired seeds, identical batches),
measure the noise floor on your data, and name a winner only if it clears that
floor. "Use dense" is a frequent and correct answer.
"""

from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional, Sequence

import torch
import torch.nn as nn

from aryanet.mechanisms import Candidate, control_candidates, default_candidates
from aryanet.nets import build_net
from aryanet.protocol import (
    Normalizer,
    TrainConfig,
    as_float_tensor,
    evaluate,
    set_seed,
    stratified_split,
    train_to_convergence,
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
        include_controls: bool = False,
        config: TrainConfig = TrainConfig(),
        val_per_class: int = 50,
        device: Optional[torch.device] = None,
        verbose: bool = True) -> SearchResult:
    """Search mechanisms on your data and return the one that actually won.

    Args:
      x_train, y_train: features and integer labels. Images as (N, C, H, W),
        tabular as (N, F). numpy or torch, uint8 or float.
      x_val, y_val: optional. If omitted, a stratified split is held out of
        training data (`val_per_class` per class) -- and note that shrinks the
        training set, which matters when labels are the scarce resource.
      x_test, y_test: optional but recommended. The winner is CHOSEN on
        validation, so its validation number is optimistically biased; a test set
        is the only way to get an honest final figure.
      seeds: paired repetitions. Below 5 the variance estimate is too unstable to
        support a conclusion; the decision rule will usually decline to call one.
      include_controls: add diagnostic arms (randk) that answer "why", not
        "which". Read architecture/skills.md before interpreting them.

    Returns a SearchResult whose `.model` is trained and ready, and whose
    `.report()` states the effect size, the noise floor, and what it could not
    resolve.
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

    arms = list(candidates) if candidates is not None else default_candidates()
    if include_controls:
        arms = arms + control_candidates()
    if not any(a.name == "dense" for a in arms):
        raise ValueError("candidates must include a 'dense' reference arm.")

    if verbose:
        print("aryanet: {} train / {} val, shape {}, {} classes".format(
            xtr.shape[0], xva.shape[0], input_shape, num_classes))
        print("         {} arms x {} seeds = {} runs, device={}\n".format(
            len(arms), seeds, len(arms) * seeds, device))

    acc: Dict[str, Dict[int, float]] = {a.name: {} for a in arms}
    cost: Dict[str, Dict[str, float]] = {}
    best_state: Dict[str, Any] = {}
    best_val: Dict[str, float] = {}
    cost_acc: Dict[str, List[Any]] = {a.name: [] for a in arms}

    for seed in range(seeds):
        for arm in arms:
            set_seed(seed)                     # identical init across arms
            model = build_net(input_shape, num_classes, arm.factory)
            r = train_to_convergence(model, xtr_n, ytr, xva_n, yva,
                                     seed, config, device)
            acc[arm.name][seed] = r.val_acc
            cost_acc[arm.name].append(r)
            if r.val_acc > best_val.get(arm.name, -1.0):
                best_val[arm.name] = r.val_acc
                best_state[arm.name] = {k: v.cpu().clone()
                                        for k, v in model.state_dict().items()}
            if verbose:
                base = acc["dense"].get(seed)
                delta = "" if base is None or arm.name == "dense" else \
                    "  ({:+5.2f})".format((r.val_acc - base) * 100)
                print("  seed {} {:<20} val={:.4f}{}  act={:5.1f}  {:4.1f}s".format(
                    seed, arm.name, r.val_acc, delta, r.active_units, r.seconds))
        if verbose:
            print()

    for name, runs in cost_acc.items():
        cost[name] = {
            "active_units": sum(r.active_units for r in runs) / len(runs),
            "dead_fraction": sum(r.dead_frac for r in runs) / len(runs),
            "seconds": sum(r.seconds for r in runs) / len(runs),
            "best_step": sum(r.best_step for r in runs) / len(runs),
        }

    stats = [paired_stats(a.name,
                          [(acc[a.name][s] - acc["dense"][s]) * 100
                           for s in range(seeds)])
             for a in arms]
    decision = decide(stats, complexity={a.name: a.complexity for a in arms})

    model = build_net(input_shape, num_classes,
                      next(a.factory for a in arms if a.name == decision.winner))
    model.load_state_dict(best_state[decision.winner])
    model.to(device).eval()

    test_acc = None
    if xte_n is not None:
        test_acc = evaluate(model, xte_n, yte, device)

    result = SearchResult(
        winner=decision.winner, decision=decision, arms=stats, model=model,
        normalizer=norm,
        per_arm_acc={n: sum(v.values()) / len(v) for n, v in acc.items()},
        cost=cost, test_acc=test_acc, n_train=int(xtr.shape[0]),
        n_val=int(xva.shape[0]), seeds=seeds)

    if verbose:
        print(result.report())
    return result


def _format_report(r: SearchResult) -> str:
    lines = []
    add = lines.append
    add("=" * 74)
    add("ARYANET SEARCH -- paired delta vs dense, percentage points")
    add("=" * 74)
    add("{} training / {} validation examples, {} seeds".format(
        r.n_train, r.n_val, r.seeds))
    add("")
    add("  {:<20}{:>9}{:>9}{:>7}{:>8}{:>10}".format(
        "arm", "val acc", "delta", "sd", "p", "signs"))
    for a in r.arms:
        if a.name == "dense":
            add("  {:<20}{:>9.4f}{:>9}{:>7}{:>8}{:>10}".format(
                a.name, r.per_arm_acc[a.name], "--", "--", "--", "(reference)"))
        else:
            add("  {:<20}{:>9.4f}{:>+9.2f}{:>7.2f}{:>8.4f}{:>10}".format(
                a.name, r.per_arm_acc[a.name], a.mean, a.sd, a.p, a.signs))

    add("")
    add("COST (what each mechanism spends, not just what it buys)")
    add("  {:<20}{:>14}{:>14}{:>12}".format(
        "arm", "active units", "dead frac", "sec/run"))
    for a in r.arms:
        c = r.cost[a.name]
        add("  {:<20}{:>14.1f}{:>14.3f}{:>12.1f}".format(
            a.name, c["active_units"], c["dead_fraction"], c["seconds"]))

    add("")
    add("-" * 74)
    if r.decision.confident:
        add("WINNER: {}".format(r.winner))
        add("  {}".format(r.decision.reason))
        if r.decision.tied_with:
            add("  Tied arms are genuinely interchangeable here -- if you prefer")
            add("  one for reasons the accuracy cannot see (inference cost, a")
            add("  standard component your team already knows), take it.")
    else:
        add("WINNER: dense (no mechanism cleared the noise floor)")
        add("  {}".format(r.decision.reason))
        add("  This is a real answer. On this data, at this label budget, the")
        add("  mechanisms tested do not beat a standard network.")

    noise = [a for a in r.arms if a.name != "dense"]
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
    add("-" * 74)
    return "\n".join(lines)
