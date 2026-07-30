"""
The architecture axis: capacity and shape as first-class search candidates.

WHY THIS EXISTS
---------------
Until now the search held one CNN fixed and varied only the activation. That
answers "which mechanism helps on your data" and quietly assumes the answer to a
bigger question -- "what should I train at all" -- by hard-coding it.

At a few hundred labels that assumption is the expensive one. Capacity is
normally the dominant knob in that regime, a SMALLER network frequently beats a
larger one outright, and multinomial logistic regression is a genuine competitor
rather than a straw man. A tool that can only return a mechanism will return a
mechanism, even when the honest answer was "use a smaller model" or "a linear
classifier was as good as anything you tried". Adding the axis is what makes
those answers reachable.

EVIDENCE STATUS -- READ THIS BEFORE TRUSTING A CAPACITY RESULT
--------------------------------------------------------------
The mechanism candidates carry real measurements (MNIST, 300-600 labels, 8
seeds, paired). The architecture candidates carry NONE. Every recorded number in
this repo comes from `channels=(16, 32), hidden=128`; the project has never swept
width or depth at any budget. These variants are here to be measured on your
data, not because anything says they help. The `evidence` string on each says so
rather than leaving the reader to infer confidence from the fact that it shipped.

THE CONFOUND THIS AXIS REINTRODUCES, ON PURPOSE
-----------------------------------------------
`architecture/skills.md` warns that a mechanism which changes parameter count
confounds "data efficiency" with "just a bigger/smaller model". The mechanism
axis was designed to avoid that -- every mechanism arm shares one architecture.
The architecture axis cannot avoid it, because model size IS the variable. So it
is reported instead: every arm logs its parameter count beside its accuracy, and
`cnn_shallow` is deliberately sized (hidden=64) to land within ~3% of the default
CNN's parameter count at 28x28 (201,578 vs 206,922), which makes it a depth
ablation at roughly matched size rather than a second width knob.

COMPLEXITY RANKS
----------------
The decision rule breaks statistical ties toward lower `complexity`, so the ranks
decide what the library recommends when two arms are indistinguishable. They are
ordered by parameter count at 28x28, which is a measurable definition of "simpler"
rather than a taste judgement:

    linear 7,850 < cnn_narrow 26,698 < cnn_shallow 201,578
                 ~ cnn 206,922 < cnn_wide 824,458

An arm's rank is `architecture.complexity * 10 + mechanism.complexity`, so the
ARCHITECTURE dominates the tie-break and the mechanism only orders arms within
one architecture. That ordering is the point: when a linear model ties the wide
CNN with kWTA, the library must hand back the linear model. The reverse would be
the exact overclaim it exists to prevent.

`pretrained` is the one deliberate exception to "rank tracks parameter count". Its
TRAINABLE footprint is tiny (~5k head params, smaller than `cnn_narrow`), but as
an artifact it is the heaviest thing here: ~11.7M frozen params, an extra
dependency, and a weights download. It is therefore ranked ABOVE `cnn_wide`
(complexity 5, the largest), so a pretrained probe that only TIES a lighter arm
loses the tie-break and is not recommended -- it must strictly win. Ranking it by
its trainable params would have made a heavyweight download the tool's default
tie-winner, which is backwards.
"""

from __future__ import annotations

from typing import Any, Callable, List, NamedTuple, Optional, Sequence

import torch.nn as nn

from scarce.mechanisms import Candidate, default_candidates
from scarce.nets import Net, build_cnn, build_linear, build_net
from scarce.pretrained import build_pretrained

#: The architecture every delta is measured against: the frozen research stack.
#: Changing this would make new results incomparable with every recorded run.
REFERENCE_ARCHITECTURE = "cnn"
REFERENCE_MECHANISM = "dense"
REFERENCE_ARM = "{}/{}".format(REFERENCE_ARCHITECTURE, REFERENCE_MECHANISM)


class Architecture(NamedTuple):
    """One model shape the search can choose.

    `builder(input_shape, num_classes, activation_fn) -> Net`, matching
    `build_net`'s signature so an architecture is a drop-in for it.

    `supports_mechanism=False` marks a shape with no hidden activation site.
    Such an architecture is paired only with `dense`; it is NOT paired with a
    mechanism and the mechanism silently dropped, which would report an arm as
    though a mechanism had been applied to it when it had not.
    """
    name: str
    builder: Callable[[Sequence[int], int, Callable[[], nn.Module]], Net]
    evidence: str
    complexity: int
    supports_mechanism: bool = True


def _cnn(channels: Sequence[int], hidden: int):
    def build(input_shape, num_classes, activation_fn=nn.ReLU):
        return build_cnn(input_shape, num_classes, activation_fn,
                         channels=channels, hidden=hidden)
    return build


def _pretrained(backbone="resnet18", image_size=64):
    """A builder that ignores `activation_fn` -- a frozen backbone has no site for
    a mechanism -- and binds the backbone choice. torchvision is imported lazily
    inside `build_pretrained`, so referencing this builder costs nothing until an
    arm is actually trained.
    """
    def build(input_shape, num_classes, activation_fn=None):
        return build_pretrained(input_shape, num_classes,
                                backbone=backbone, image_size=image_size)
    return build


def _reference_cnn(input_shape, num_classes, activation_fn=nn.ReLU):
    """The frozen stack, routed through `build_net` itself.

    `build_cnn` with default arguments is bitwise identical to this (asserted in
    tests), but calling `build_net` directly means the reference arm can never
    drift from the function the recorded results and the live CIFAR run use, even
    if `build_cnn` is later changed.
    """
    return build_net(input_shape, num_classes, activation_fn)


LINEAR = Architecture(
    "linear", build_linear,
    "Multinomial logistic regression on raw inputs, ~7.9k params at 28x28. "
    "UNMEASURED in this repo. Included because at a few hundred labels it is a "
    "real competitor and costs seconds to train -- a search that cannot lose to "
    "logistic regression is not measuring anything.",
    0, supports_mechanism=False)

CNN_NARROW = Architecture(
    "cnn_narrow", _cnn((8, 16), 32),
    "Half-width conv stack, hidden=32; ~26.7k params, 7.8x smaller than the "
    "default. UNMEASURED. The variant most likely to win at the scarcest "
    "budgets, where capacity is usually the binding constraint.",
    1)

CNN_SHALLOW = Architecture(
    "cnn_shallow", _cnn((16,), 64),
    "One conv block instead of two, hidden=64; ~201.6k params, within 3% of the "
    "default's 206.9k at 28x28. UNMEASURED. Sized to isolate DEPTH at roughly "
    "matched parameter count rather than to be a second width knob.",
    2)

CNN = Architecture(
    "cnn", _reference_cnn,
    "The frozen research stack: conv 16/32, hidden 128, ~206.9k params. Every "
    "recorded number in this project comes from this shape, which is why it is "
    "the reference arm and not because it is known to be the right size.",
    3)

CNN_WIDE = Architecture(
    "cnn_wide", _cnn((32, 64), 256),
    "Double-width conv stack, hidden=256; ~824.5k params, 4x the default. "
    "UNMEASURED. Present so the search can find that more capacity helps -- "
    "which is a possible answer, and one a low-data prior would not predict.",
    4)


PRETRAINED = Architecture(
    "pretrained", _pretrained("resnet18"),
    "Frozen ImageNet resnet18 backbone + a fresh linear head (a LINEAR PROBE, not "
    "fine-tuning): only the ~5k-param head trains, the ~11.7M backbone params are "
    "frozen. Needs torchvision (optional extra `scarce[pretrained]`) and a one-time "
    "weights download. UNMEASURED in this repo, but at a few hundred real-world "
    "images a pretrained probe is usually the strongest baseline -- included so "
    "the search can recommend it, and can honestly LOSE to it.",
    5, supports_mechanism=False)


def default_architectures() -> List[Architecture]:
    """The five lightweight core shapes, simplest first.

    Deliberately excludes `PRETRAINED`: these are torchvision-free, download-free,
    and tabular-capable, and every existing test iterates this list on the
    assumption that building any of them is cheap and offline. `PRETRAINED` breaks
    all three, so it lives in the `"full"` space (see `full_architectures`) and is
    opt-in, never a silent dependency of the default search.
    """
    return [LINEAR, CNN_NARROW, CNN_SHALLOW, CNN, CNN_WIDE]


def full_architectures() -> List[Architecture]:
    """The core shapes plus the pretrained-backbone probe.

    This is what `architectures="full"` searches. `PRETRAINED` is appended last
    and ranked highest-complexity, so on an unresolvable tie the decision rule
    prefers any of the lighter arms -- a pretrained probe has to strictly BEAT the
    field to be recommended, never merely match it.
    """
    return default_architectures() + [PRETRAINED]


class Arm(NamedTuple):
    """One (architecture x mechanism) cell -- what actually gets trained.

    `build(input_shape, num_classes) -> Net` has the architecture and the
    mechanism already bound, so the search loop stays identical whether it is
    running one architecture or five.
    """
    name: str
    architecture: str
    mechanism: str
    build: Callable[[Sequence[int], int], Net]
    complexity: int
    evidence: str
    is_reference: bool = False


def _bind(arch: Architecture, mech: Candidate) -> Arm:
    name = "{}/{}".format(arch.name, mech.name)
    return Arm(
        name=name,
        architecture=arch.name,
        mechanism=mech.name,
        build=(lambda shape, nc, _a=arch, _m=mech:
               _a.builder(shape, nc, _m.factory)),
        complexity=arch.complexity * 10 + mech.complexity,
        evidence="{} | {}".format(arch.evidence, mech.evidence),
        is_reference=(arch.name == REFERENCE_ARCHITECTURE
                      and mech.name == REFERENCE_MECHANISM),
    )


def cross(architectures: Sequence[Architecture],
          mechanisms: Sequence[Candidate]) -> List[Arm]:
    """Full (architecture x mechanism) product, minus the cells that do not exist.

    An architecture with `supports_mechanism=False` appears once, paired with
    `dense`. The reference cell (`cnn/dense`) is added even if the caller left
    the default CNN or the dense mechanism out of the lists -- every delta is
    measured against it, so the search is meaningless without it and silently
    substituting a different reference would make the numbers incomparable with
    every recorded run.
    """
    dense = next((m for m in mechanisms if m.name == REFERENCE_MECHANISM), None)
    if dense is None:
        dense = default_candidates()[0]

    arms: List[Arm] = []
    for arch in architectures:
        if arch.supports_mechanism:
            for mech in mechanisms:
                arms.append(_bind(arch, mech))
        else:
            arms.append(_bind(arch, dense))

    if not any(a.is_reference for a in arms):
        arms.insert(0, _bind(CNN, dense))
    return _reference_first(arms)


def default_arms() -> List[Arm]:
    """The curated default space: 12 arms, not the 17-cell full product.

    The full cross of five architectures and four mechanisms is 17 trainable
    cells, and most of the extra cells answer nothing anyone asked -- kWTA at
    k=0.10 on the wide stack is not a question, it is a cell. The curated space
    spends its arms on the two contrasts that matter:

      * capacity swept end to end under `dense` (5 arms), which is the axis
        with no evidence behind it at all;
      * mechanisms crossed with three capacities (7 arms), enough to see whether
        a mechanism's effect survives a change of model size -- if kWTA only
        helps at one width, that is a fact about the width, not the mechanism.

    Use `cross(default_architectures(), default_candidates())` for the full
    product when you have the compute; `fit(..., budget=...)` for when you do
    not.
    """
    mechs = {c.name: c for c in default_candidates()}
    dense = mechs["dense"]
    dropout = mechs["dropout_k0.2"]
    kwta = mechs["kwta_channel_k0.2"]
    kwta_lo = mechs["kwta_channel_k0.1"]

    pairs = [
        (LINEAR, dense),
        (CNN_NARROW, dense), (CNN_NARROW, dropout), (CNN_NARROW, kwta),
        (CNN_SHALLOW, dense),
        (CNN, dense), (CNN, dropout), (CNN, kwta), (CNN, kwta_lo),
        (CNN_WIDE, dense), (CNN_WIDE, dropout), (CNN_WIDE, kwta),
    ]
    return _reference_first([_bind(a, m) for a, m in pairs])


def arms_from_candidates(candidates: Sequence[Candidate]) -> List[Arm]:
    """Mechanism-only arms on the frozen stack -- the original search space.

    Names are the bare mechanism names (`dense`, `kwta_channel_k0.2`), NOT
    `cnn/dense`, so a mechanism-only search reports exactly what it always
    reported and existing callers and recorded outputs stay comparable.
    """
    arms = [Arm(name=c.name, architecture=REFERENCE_ARCHITECTURE,
                mechanism=c.name,
                build=(lambda shape, nc, _c=c: build_net(shape, nc, _c.factory)),
                complexity=c.complexity,
                evidence=c.evidence,
                is_reference=(c.name == REFERENCE_MECHANISM))
            for c in candidates]
    return _reference_first(arms)


def _reference_first(arms: Sequence[Arm]) -> List[Arm]:
    """Reference arm first, so the per-seed delta is printable as runs complete."""
    ref = [a for a in arms if a.is_reference]
    return ref + [a for a in arms if not a.is_reference]


def resolve_arms(candidates: Optional[Sequence[Candidate]],
                 architectures: Any = None,
                 controls: Sequence[Candidate] = ()) -> List[Arm]:
    """Turn `fit`'s space arguments into the list of arms to train.

    `architectures` accepts:
      None        -- mechanism-only on the frozen stack. The original search
                     space, and still the default, so no existing caller's
                     results change under them.
      "default"   -- the curated 12-arm (architecture x mechanism) space.
      "full"      -- the complete cross product of every architecture with
                     every mechanism, PLUS the frozen pretrained-backbone probe
                     (18 cells with the defaults). The pretrained arm pulls in
                     torchvision and a one-time weights download, so it is reached
                     here and by an explicit list, never through `None`/`"default"`.
      a sequence  -- cross those architectures with the mechanisms.
    """
    mechs = list(candidates) if candidates is not None else default_candidates()
    mechs = mechs + list(controls)

    if architectures is None:
        return arms_from_candidates(mechs)
    if isinstance(architectures, str):
        key = architectures.lower()
        if key == "default":
            # Controls are diagnostics for the mechanism axis, so they attach to
            # the reference architecture only -- crossing them with capacity
            # would multiply arms to answer a question nobody asked.
            return _reference_first(default_arms()
                                    + [_bind(CNN, c) for c in controls])
        if key == "full":
            return cross(full_architectures(), mechs)
        raise ValueError(
            "architectures must be None, 'default', 'full', or a sequence of "
            "Architecture; got {!r}".format(architectures))
    return cross(list(architectures), mechs)
