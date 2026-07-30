"""Unit tests for the pretrained-backbone linear-probe arm.

torchvision is an OPTIONAL dependency, so the whole module skips cleanly where it
is absent (`pytest.importorskip`) -- but where it is present these DO exercise the
real torchvision code path: a real resnet18 is constructed, frozen, adapted, and
trained one step. To stay offline and fast they build the backbone with
`weights=None` (random-but-frozen weights, no ~45MB download), which exercises
every line that matters -- freezing, the input adapter, frozen BatchNorm, gradient
flow -- without the network. The `PRETRAINED` architecture used in production
pulls real ImageNet weights; that download is not a unit-test concern.

Every test that draws random data seeds first (tests/skills.md: an unseeded draw
fails on a schedule you do not control).
"""

import pytest
import torch
import torch.nn as nn

pytest.importorskip("torchvision")

import scarce
from scarce.architectures import (
    CNN,
    LINEAR,
    PRETRAINED,
    Architecture,
    cross,
    default_architectures,
    full_architectures,
    resolve_arms,
)
from scarce.mechanisms import default_candidates
from scarce.nets import Net, count_parameters
from scarce.pretrained import build_pretrained
from scarce.protocol import TrainConfig


# A small, offline, frozen-random backbone. image_size=32 is resnet's minimum and
# the cheapest option, which keeps these tests quick on CPU.
def _probe(input_shape, num_classes):
    return build_pretrained(input_shape, num_classes, weights=None, image_size=32)


# ------------------------------ the module builds -----------------------------

def test_builder_returns_a_net_with_the_three_expected_submodules():
    net = _probe((1, 8, 8), 3)
    assert isinstance(net, Net)
    assert hasattr(net, "features") and hasattr(net, "classifier")
    assert hasattr(net, "penultimate")


@pytest.mark.parametrize("shape", [(1, 8, 8), (3, 20, 20), (1, 28, 28), (3, 32, 32)])
def test_forward_shape_on_grayscale_and_rgb(shape):
    """Grayscale (1,H,W) and RGB (3,H,W), small and large, all produce logits."""
    torch.manual_seed(0)
    net = _probe(shape, 4)
    out = net(torch.randn(5, *shape))
    assert out.shape == (5, 4), shape


def test_penultimate_is_hookable_and_is_the_pooled_feature_vector():
    """protocol.representation_stats hooks `.penultimate`; it must fire and expose
    the pooled frozen-feature vector (resnet18 -> 512-d), so the diagnostics
    report the code the linear head actually reads.
    """
    torch.manual_seed(0)
    net = _probe((1, 8, 8), 3)
    seen = []
    h = net.penultimate.register_forward_hook(lambda m, i, o: seen.append(o.shape))
    net(torch.randn(2, 1, 8, 8))
    h.remove()
    assert seen and tuple(seen[0]) == (2, 512)


# ------------------------------ frozen vs trainable ---------------------------

def test_backbone_is_frozen_and_only_the_head_is_trainable():
    net = _probe((3, 32, 32), 3)
    assert all(not p.requires_grad for p in net.features.backbone.parameters())
    head = net.classifier[1]
    assert isinstance(head, nn.Linear)
    assert all(p.requires_grad for p in head.parameters())


def test_count_parameters_reports_only_the_trainable_head():
    """The parameter-cost column must not silently include ~11.7M frozen params."""
    net = _probe((1, 8, 8), 7)
    assert count_parameters(net) == 512 * 7 + 7          # resnet18 feature dim


def test_one_training_step_updates_only_the_head():
    torch.manual_seed(0)
    net = _probe((1, 8, 8), 3)
    net.train()
    x = torch.randn(16, 1, 8, 8)
    y = torch.randint(0, 3, (16,))

    bb_before = torch.cat(
        [p.detach().flatten() for p in net.features.backbone.parameters()]).clone()
    head_before = net.classifier[1].weight.detach().clone()

    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    opt.zero_grad()
    nn.CrossEntropyLoss()(net(x), y).backward()
    opt.step()

    bb_after = torch.cat(
        [p.detach().flatten() for p in net.features.backbone.parameters()])
    assert torch.equal(bb_before, bb_after)              # backbone untouched
    assert not torch.equal(head_before, net.classifier[1].weight.detach())
    assert all(p.grad is None for p in net.features.backbone.parameters())
    assert net.classifier[1].weight.grad is not None


def test_frozen_batchnorm_stays_in_eval_through_train():
    """A frozen feature extractor must keep BN running stats fixed even while the
    head trains, or the 'frozen' features drift with batch composition.
    """
    torch.manual_seed(0)
    net = _probe((1, 8, 8), 3)
    net.train()
    bn = next(m for m in net.features.backbone.modules()
              if isinstance(m, nn.BatchNorm2d))
    assert not bn.training                               # eval despite net.train()
    mean_before = bn.running_mean.clone()
    net(torch.randn(16, 1, 8, 8))
    assert torch.equal(mean_before, bn.running_mean)     # running stats frozen


# ------------------------------ argument handling -----------------------------

def test_flat_tabular_input_is_refused_with_a_clear_message():
    with pytest.raises(ValueError, match="image input"):
        build_pretrained((20,), 3, weights=None)


def test_unknown_backbone_is_refused_not_guessed():
    with pytest.raises(ValueError, match="not supported"):
        build_pretrained((3, 32, 32), 3, backbone="not_a_model", weights=None)


def test_activation_fn_is_accepted_and_ignored():
    """A frozen backbone has no activation site; passing one must not change the
    model, so the builder stays a drop-in for the Architecture interface.
    """
    torch.manual_seed(0)
    a = build_pretrained((1, 8, 8), 3, weights=None, image_size=32,
                         activation_fn=nn.Tanh)
    torch.manual_seed(0)
    b = build_pretrained((1, 8, 8), 3, weights=None, image_size=32)
    assert count_parameters(a) == count_parameters(b)


# ------------------------------ the search space ------------------------------

def test_pretrained_declares_it_has_no_mechanism_site():
    assert PRETRAINED.supports_mechanism is False


def test_pretrained_is_never_paired_with_a_mechanism_it_cannot_apply():
    """Mirror of the LINEAR guarantee (test_architectures.py): a frozen backbone
    has no hidden activation site, so pairing it with kWTA/dropout and dropping
    the mechanism would report an arm as though a mechanism had been applied.
    """
    for arms in (resolve_arms(None, "full"),
                 cross(full_architectures(), default_candidates())):
        pre = [a for a in arms if a.architecture == "pretrained"]
        assert len(pre) == 1
        assert pre[0].mechanism == "dense"


def test_full_space_includes_pretrained_and_default_space_does_not():
    """Opt-in: `"full"` reaches it, `"default"` (and `None`) never pull torchvision
    or a download in silently.
    """
    full_names = {a.architecture for a in resolve_arms(None, "full")}
    assert "pretrained" in full_names
    for space in (resolve_arms(None, None), resolve_arms(None, "default")):
        assert "pretrained" not in {a.architecture for a in space}
    # the lightweight core list is unchanged
    assert "pretrained" not in {a.name for a in default_architectures()}


def test_pretrained_outranks_every_lighter_arm_on_a_tie():
    """It must be LEAST-preferred: a probe that only ties a lighter arm should lose
    the tie-break. Arm complexity is arch*10 + mech, and pretrained is arch rank 5.
    """
    rank = {a.name: a.complexity for a in resolve_arms(None, "full")}
    others = [v for k, v in rank.items() if k != "pretrained/dense"]
    assert rank["pretrained/dense"] > max(others)


# ------------------------------ end to end, tiny ------------------------------

def _toy(n=150, classes=3, seed=0):
    torch.manual_seed(seed)
    w = torch.randn(classes, 1, 8, 8)
    y = torch.randint(0, classes, (n,))
    return w[y] + 1.5 * torch.randn(n, 1, 8, 8), y


# An offline pretrained arm (random frozen weights, no download) so `fit` can be
# exercised with the pretrained arm competing without touching the network.
_OFFLINE_PRETRAINED = Architecture(
    "pretrained",
    lambda s, nc, a=None: build_pretrained(s, nc, weights=None, image_size=32),
    PRETRAINED.evidence, PRETRAINED.complexity, supports_mechanism=False)

TINY = TrainConfig(max_steps=30, eval_every=15, patience=2, batch_size=32)


def test_fit_runs_the_pretrained_arm_reports_it_and_renders():
    """The pretrained arm trains inside the real search loop, appears in the
    reported arms and the cost table, and the report footnotes that the backbone
    is frozen and uncounted.
    """
    x, y = _toy()
    r = scarce.fit(x, y, architectures=[CNN, LINEAR, _OFFLINE_PRETRAINED],
                   seeds=2, config=TINY, val_per_class=15, verbose=False)
    names = {a.name for a in r.arms}
    assert "pretrained/dense" in names
    assert "pretrained/dense" in r.cost
    # only the trainable head is counted in the cost column
    assert r.cost["pretrained/dense"]["params"] == 512 * 3 + 3
    text = r.report()
    assert "pretrained/dense" in text
    assert "FROZEN and NOT counted" in text


def test_fit_can_hand_back_the_pretrained_model_and_predict():
    """If the pretrained arm wins, the returned model must be that architecture and
    able to predict -- the arch axis makes a skeleton/weights mismatch a real risk.
    """
    x, y = _toy()
    r = scarce.fit(x, y, architectures=[CNN, LINEAR, _OFFLINE_PRETRAINED],
                   seeds=2, config=TINY, val_per_class=15, verbose=False)
    assert r.predict(x[:4]).shape == (4,)
