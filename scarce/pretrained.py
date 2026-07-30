"""
Pretrained-backbone linear probe -- the baseline the search could not previously try.

WHY THIS EXISTS
---------------
For someone with a few hundred real-world images, the genuinely strongest move is
usually a PRETRAINED backbone: freeze an ImageNet feature extractor and train a
linear head on top of its features. Until now `scarce.fit` searched activation
mechanisms and small-CNN capacities only, so it could neither recommend that move
nor honestly LOSE to it. The whole point of the library is "measure, don't
assume" -- and a search that cannot try the thing that usually wins is assuming
it away. This arm makes the tool able to return "a pretrained linear probe beat
everything you tried."

WHAT IT IS, PRECISELY
---------------------
A LINEAR PROBE, not fine-tuning:

  * every backbone parameter has `requires_grad=False`;
  * the ImageNet classification head is discarded and replaced with a fresh
    `nn.Linear(feature_dim, num_classes)`;
  * ONLY that head trains.

This is what makes it CPU-affordable (the backbone is forward-only, run under
`torch.no_grad()`) and it is the canonical low-data baseline. Fine-tuning the
backbone would be a different, far more expensive experiment and is deliberately
not what this arm measures.

THE INPUT ADAPTER (part of `forward`)
-------------------------------------
ImageNet backbones expect 3xHxW, ImageNet-normalized input. The adapter handles:

  * grayscale (1,H,W) -> repeated to 3 channels;
  * small images -> bilinearly resized up to `image_size` (resnet tolerates any
    size >= 32 via its adaptive pool). We default to 64, NOT 224: 224 is
    ImageNet-native and gives the best features but is far too slow on CPU; 32 is
    the cheapest and CIFAR-native; 64 is a modest middle that keeps more spatial
    detail than 32 at a fraction of 224's cost. The backbone runs forward-only, so
    this is one un-differentiated forward per step.
  * ImageNet mean/std applied to the 3-channel input.

A HONEST CAVEAT ON NORMALIZATION. `scarce.fit` standardizes every arm's input
with a per-dataset `Normalizer` fitted on the training split (shared across arms,
so a difference is never a preprocessing difference). By the time the input
reaches this module it is already standardized, so the ImageNet mean/std the
adapter applies does not exactly reconstruct ImageNet preprocessing. That is
fine for a probe: the transform is deterministic and identical on every forward,
so the frozen features are consistent, and the trainable linear head adapts to
whatever consistent feature distribution results -- which is exactly what a linear
probe does. Perfect fidelity would feed [0,1] images and let this adapter own all
normalization; the shared pipeline makes that a documented approximation, not a
bug.

FROZEN BATCHNORM. resnet carries BatchNorm layers. A frozen feature extractor
must keep them in eval mode (ImageNet running statistics) even when the head is
training, or the "frozen" features would drift with batch composition. The
backbone is forced into `.eval()` and kept there through `.train()`.

COST, STATED PLAINLY
--------------------
`count_parameters` reports only the TRAINABLE head (~5k params for resnet18 ->
10 classes). The ~11.7M frozen backbone params, the extra torchvision dependency,
and the one-time weights download are the real cost, which is why the arm is
ranked LEAST-preferred on a tie (see `architectures.PRETRAINED`) and why the
`fit` report footnotes that the backbone is frozen and uncounted.

FEATURE CACHING IS DEFERRED, ON PURPOSE. The backbone is frozen, so its features
are constant across seeds and epochs -- extracting them once and training the head
on the cache would be dramatically cheaper than re-running the backbone every
step. Doing that cleanly means threading a cache through `protocol.train_to_
convergence` and `search.fit`, which is invasive and out of scope here. Instead
the backbone is frozen and wrapped in `torch.no_grad()` (so no autograd graph is
built through millions of params, the bulk of the saving) and the cross-seed
feature cache is left as the obvious next optimization.

torchvision is a LAZY, OPTIONAL dependency: it is imported INSIDE `build_pretrained`,
never at module load, so `import scarce` and the core `fit()` paths work with
torchvision absent. When it is missing this raises a clear, actionable error
naming the `scarce[pretrained]` extra.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from scarce.nets import Net

#: ImageNet channel statistics, the distribution every torchvision backbone was
#: trained on. Applied by the adapter in `forward`.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Resize target. 32 = cheapest / CIFAR-native / resnet minimum; 224 = ImageNet-
#: native / best features / too slow on CPU; 64 = the documented middle default.
DEFAULT_IMAGE_SIZE = 64

#: Backbones with a `.fc` head this builder knows how to strip. resnet-family
#: only, kept deliberately small -- the arm exists to be measured, not to be a
#: model zoo.
SUPPORTED_BACKBONES = ("resnet18", "resnet34", "resnet50")


class _FrozenBackbone(nn.Module):
    """Input adapter + frozen ImageNet backbone -> pooled feature vector.

    Emits the pooled, flattened feature vector the linear head reads. Kept in
    eval mode through `.train()` so its BatchNorm running statistics never move,
    and its forward runs under `torch.no_grad()` so no autograd graph is built
    through the frozen weights (the head still trains: it reads a detached
    feature tensor and its own parameters carry the gradient).
    """

    def __init__(self, backbone: nn.Module, image_size: int,
                 mean: Sequence[float], std: Sequence[float]):
        super().__init__()
        self.backbone = backbone
        self.image_size = int(image_size)
        self.register_buffer(
            "mean", torch.tensor(mean, dtype=torch.float32).reshape(1, 3, 1, 1))
        self.register_buffer(
            "std", torch.tensor(std, dtype=torch.float32).reshape(1, 3, 1, 1))

    def train(self, mode: bool = True) -> "_FrozenBackbone":
        # The head (elsewhere in the Net) follows `mode`; the backbone never does.
        super().train(mode)
        self.backbone.eval()
        return self

    def _adapt(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                "the pretrained backbone needs image input (N, C, H, W); got a "
                "tensor with {} dims. It cannot be used on flat/tabular features -- "
                "that is why `PRETRAINED` is not in the tabular-capable default "
                "architecture set.".format(x.dim()))
        c = x.shape[1]
        if c == 1:
            x = x.repeat(1, 3, 1, 1)          # grayscale -> 3 channels
        elif c != 3:
            raise ValueError(
                "the pretrained backbone expects 1 (grayscale) or 3 (RGB) input "
                "channels; got {}.".format(c))
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                              mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._adapt(x)
        self.backbone.eval()                  # defensive: frozen BN, always
        with torch.no_grad():
            feats = self.backbone(x)
        return feats


def build_pretrained(input_shape: Sequence[int],
                     num_classes: int,
                     backbone: str = "resnet18",
                     weights: Optional[str] = "DEFAULT",
                     image_size: int = DEFAULT_IMAGE_SIZE,
                     activation_fn: Optional[object] = None) -> Net:
    """A frozen ImageNet backbone with a fresh trainable linear head.

    Returns a `scarce.nets.Net` whose `.features` is the frozen backbone (+ input
    adapter), `.classifier` is the trainable linear head, and `.penultimate` is
    the pooled frozen-feature vector the head reads -- so the existing diagnostics
    in `protocol.representation_stats`, which hook `.penultimate`, report the
    feature-code statistics unchanged.

    Args:
      input_shape: (C, H, W), excluding the batch dim. Images only -- a flat
        (F,) shape raises, since an ImageNet backbone has nothing to do with
        tabular features.
      num_classes: size of the fresh linear head.
      backbone: one of `SUPPORTED_BACKBONES`.
      weights: torchvision weights spec. "DEFAULT" downloads ImageNet weights
        (the point of the arm); None gives a randomly-initialized-but-still-frozen
        backbone, used by the tests to exercise the real code path offline.
      image_size: the square size inputs are resized to (see module docstring).
      activation_fn: accepted for builder-interface compatibility and IGNORED --
        a frozen backbone has no activation site for a mechanism to act on, which
        is why `architectures.PRETRAINED` declares `supports_mechanism=False`.

    torchvision is imported HERE, lazily, so `import scarce` works without it.
    """
    shape = tuple(int(s) for s in input_shape)
    if len(shape) != 3:
        raise ValueError(
            "the pretrained backbone needs image input (C, H, W); got {}. "
            "Reshape images to (C, H, W), or use a CNN/linear arm for flat "
            "features.".format(shape))

    try:
        from torchvision import models
    except ImportError as exc:                # pragma: no cover - env dependent
        raise ImportError(
            "the pretrained-backbone arm needs torchvision, which is an OPTIONAL "
            "dependency and is not installed. Install it with:\n"
            "    pip install \"scarce[pretrained]\"\n"
            "(or `pip install torchvision`). Core `scarce.fit()` runs without "
            "it -- only this arm requires it.") from exc

    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(
            "backbone {!r} is not supported; choose one of {}. (Only resnet-"
            "family heads are stripped here.)".format(
                backbone, list(SUPPORTED_BACKBONES)))

    net = getattr(models, backbone)(weights=weights)
    feature_dim = int(net.fc.in_features)
    net.fc = nn.Identity()                    # expose pooled features, drop head
    for p in net.parameters():
        p.requires_grad_(False)               # freeze the whole backbone
    net.eval()

    features = _FrozenBackbone(net, image_size, IMAGENET_MEAN, IMAGENET_STD)
    penultimate = nn.Identity()               # the pooled feature vector, hookable
    head = nn.Linear(feature_dim, num_classes)  # the ONLY trainable parameters
    classifier = nn.Sequential(penultimate, head)
    return Net(features, classifier, penultimate)
