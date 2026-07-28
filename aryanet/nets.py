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
"""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import torch
import torch.nn as nn

HIDDEN = 128


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


def _conv_stack(shape: Tuple[int, int, int],
                activation_fn: Callable[[], nn.Module]) -> Tuple[nn.Module, int]:
    """Two conv blocks; pool only while the spatial dims are large enough.

    baseline.py pools unconditionally because 28x28 always survives two halvings.
    An 8x8 input would not, so pooling is made conditional -- the only structural
    departure from the frozen baseline, and it is a no-op at 28x28.
    """
    c_in, h, w = shape
    layers = []
    for c_out in (16, 32):
        layers += [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1), activation_fn()]
        if min(h, w) >= 4:
            layers.append(nn.MaxPool2d(2))
            h, w = h // 2, w // 2
        c_in = c_out

    features = nn.Sequential(*layers)
    with torch.no_grad():
        flat = int(features(torch.zeros(1, *shape)).flatten(1).shape[1])
    return features, flat
