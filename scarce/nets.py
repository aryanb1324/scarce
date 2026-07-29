"""
Shape-adaptive networks, structurally faithful to the research baseline.

`architecture/baseline.py` is frozen at 1x28x28 because every recorded result
depends on it. A library cannot be: users arrive with 3x32x32 images, 64x64
images, or flat feature vectors. `build_net` keeps the *structure* the research
validated -- two conv blocks, one hidden FC layer, an activation factory swapped
in at every activation site -- while adapting the input stem and the flatten
width to whatever the data actually is.

The activation-factory design is the load-bearing part. It is what makes the
comparison honest: dense and every sparse arm share one architecture and differ
only in the activation, so an accuracy difference cannot be a model-size
difference (architecture/skills.md, "Common Mistakes").

CAPACITY IS ALSO AN AXIS
------------------------
Holding the architecture fixed and varying only the activation answers "which
mechanism helps", which is not the same question as "what should I train on 300
labels". At a few hundred labels, capacity is usually the dominant knob and a
SMALLER model frequently wins outright -- so a tool that cannot vary it will
recommend a mechanism when it should have recommended a smaller net, or a linear
model.

`build_cnn` and `build_linear` add that axis. `build_net` is UNCHANGED and
untouched: recorded results and a live CIFAR run import it, so it keeps its exact
signature, structure, and parameter-initialization order. `build_cnn`'s defaults
reproduce it layer for layer and draw the same RNG in the same sequence -- which
is asserted in `tests/test_architectures.py`, not assumed, because a difference
there would silently unpair the seeds.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import torch
import torch.nn as nn

HIDDEN = 128

#: Channel widths of the frozen research stack. `build_cnn` defaults to these so
#: that `build_cnn(...)` and `build_net(...)` are the same network.
CHANNELS = (16, 32)


class Net(nn.Module):
    """Feature extractor + classifier, with the penultimate activation exposed.

    `penultimate` is the module the final linear layer reads. Diagnostics hook it
    to measure what a mechanism actually does to the representation, which the
    project requires of every mechanism (root skills.md rule 5).
    """

    def __init__(self, features: nn.Module, classifier: nn.Module,
                 penultimate: nn.Module):
        super().__init__()
        self.features = features
        self.classifier = classifier
        self.penultimate = penultimate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_net(input_shape: Sequence[int],
              num_classes: int,
              activation_fn: Callable[[], nn.Module] = nn.ReLU) -> Net:
    """Build a net for `input_shape`, excluding the batch dimension.

    (C, H, W) -> the conv architecture from `baseline.py`, with pooling applied
    only while the spatial dims can afford it, so small images do not collapse.
    (F,)      -> a two-hidden-layer MLP, for tabular data.
    """
    shape = tuple(int(s) for s in input_shape)

    if len(shape) == 3:
        features, flat = _conv_stack(shape, activation_fn)
    elif len(shape) == 1:
        features, flat = nn.Identity(), shape[0]
    else:
        raise ValueError(
            "input_shape must be (C, H, W) for images or (F,) for flat features; "
            "got {}. Reshape the data or pass images as (C, H, W).".format(shape))

    act = activation_fn()
    classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(flat, HIDDEN),
        act,
        nn.Linear(HIDDEN, num_classes),
    )
    return Net(features, classifier, act)


def build_cnn(input_shape: Sequence[int],
              num_classes: int,
              activation_fn: Callable[[], nn.Module] = nn.ReLU,
              channels: Sequence[int] = CHANNELS,
              hidden: int = HIDDEN) -> Net:
    """`build_net` with the capacity opened up: conv widths and FC width.

    `channels` also sets the DEPTH -- one entry is one conv block -- so a
    single-element tuple gives a shallower net. With the defaults this is
    `build_net` exactly: same modules, constructed in the same order, so a seeded
    init draws identical parameters.

    Capacity here is unmeasured territory for this project. Every recorded number
    comes from `channels=(16, 32), hidden=128`; nothing in the repo says a wider
    or narrower stack helps or hurts at any label budget. That is precisely why
    it is a search axis rather than a new default.
    """
    shape = tuple(int(s) for s in input_shape)
    chans = tuple(int(c) for c in channels)
    if not chans:
        raise ValueError("channels must name at least one conv block")

    if len(shape) == 3:
        features, flat = _conv_stack(shape, activation_fn, chans)
    elif len(shape) == 1:
        features, flat = nn.Identity(), shape[0]
    else:
        raise ValueError(
            "input_shape must be (C, H, W) for images or (F,) for flat features; "
            "got {}. Reshape the data or pass images as (C, H, W).".format(shape))

    act = activation_fn()
    classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(flat, hidden),
        act,
        nn.Linear(hidden, num_classes),
    )
    return Net(features, classifier, act)


def build_linear(input_shape: Sequence[int],
                 num_classes: int,
                 activation_fn: Optional[Callable[[], nn.Module]] = None) -> Net:
    """Multinomial logistic regression: flatten, one linear layer, done.

    Worth taking seriously rather than including as a foil. With a few hundred
    labels a linear model on raw pixels is a real competitor, it has ~7.9k
    parameters against the CNN's ~207k at 28x28, and it trains in seconds. A tool
    that cannot report "logistic regression was as good as anything you tried"
    is not measuring, it is advertising.

    `activation_fn` is accepted for builder-interface compatibility and IGNORED:
    there is no hidden layer, so there is no site for an activation mechanism to
    act on. `architectures.LINEAR` therefore declares `supports_mechanism=False`
    and the cross product never pairs it with one -- rather than silently
    dropping the mechanism and reporting the arm as though it had been applied.

    `penultimate` is the identity on the flattened input, which is literally the
    representation the classifier reads. The diagnostics then report input
    sparsity, which is the honest reading of "active units" for this model.
    """
    shape = tuple(int(s) for s in input_shape)
    if len(shape) not in (1, 3):
        raise ValueError(
            "input_shape must be (C, H, W) for images or (F,) for flat features; "
            "got {}.".format(shape))
    flat = 1
    for s in shape:
        flat *= s

    act = nn.Identity()
    classifier = nn.Sequential(act, nn.Linear(flat, num_classes))
    return Net(nn.Flatten(), classifier, act)


def count_parameters(model: nn.Module) -> int:
    """Trainable parameters -- the cost side of the ledger for a capacity change.

    An architecture axis reintroduces exactly the confound `architecture/skills.md`
    warns about ("a mechanism changes effective parameter count, confounding a
    data-efficiency comparison with a bigger/smaller-model comparison"). It cannot
    be designed away here, because model size IS the variable. It can be reported,
    so the reader sees which kind of difference they are looking at.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _conv_stack(shape: Tuple[int, int, int],
                activation_fn: Callable[[], nn.Module],
                channels: Sequence[int] = CHANNELS) -> Tuple[nn.Module, int]:
    """One conv block per entry in `channels`; pool while the dims can afford it.

    baseline.py pools unconditionally because 28x28 always survives two halvings.
    An 8x8 input would not, so pooling is made conditional -- the only structural
    departure from the frozen baseline, and it is a no-op at 28x28.

    `channels` defaults to the frozen (16, 32), so the two-argument call made by
    `build_net` is unchanged in every respect including RNG consumption.
    """
    c_in, h, w = shape
    layers = []
    for c_out in channels:
        layers += [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1), activation_fn()]
        if min(h, w) >= 4:
            layers.append(nn.MaxPool2d(2))
            h, w = h // 2, w // 2
        c_in = c_out

    features = nn.Sequential(*layers)
    with torch.no_grad():
        flat = int(features(torch.zeros(1, *shape)).flatten(1).shape[1])
    return features, flat
