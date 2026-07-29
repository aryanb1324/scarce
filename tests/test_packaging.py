"""Guards for the installable-package boundary.

`scarce` must install as a self-contained package: `pip install scarce` should add
ONLY `scarce` to the environment, never top-level `architecture` or `data` (very
generic names that collide with a user's own modules). To achieve that, the three
mechanism modules the library needs at runtime are vendored under
`scarce/_modules/` rather than imported from the research tree.

Vendoring means two physical copies, which invites silent drift. These tests are
the drift guard: they assert the vendored copies still match the frozen
`architecture/modules/` originals, so a change to the research module that is not
mirrored here fails CI instead of shipping a stale mechanism.
"""
import os

import scarce

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "architecture", "modules")
_VENDORED = os.path.join(_REPO, "scarce", "_modules")


def _read(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return f.read()


def test_kwta_and_randk_are_byte_identical_copies():
    """These two are copied verbatim; any divergence is drift, not intent."""
    for name in ("kwta.py", "randk.py"):
        assert _read(_VENDORED, name) == _read(_SRC, name), name


def test_kwta_v2_differs_only_in_its_one_internal_import():
    """kwta_v2 is verbatim except the single import repointed into scarce.

    Normalizing that one known line back must reproduce the original exactly, so
    the test documents the ONLY sanctioned difference and catches everything else.
    """
    vendored = _read(_VENDORED, "kwta_v2.py")
    original = _read(_SRC, "kwta_v2.py")
    normalized = vendored.replace(
        "from scarce._modules.kwta import KWinnersTakeAll",
        "from architecture.modules.kwta import KWinnersTakeAll",
    )
    assert normalized == original


def test_vendored_modules_never_import_the_research_tree():
    """The point of vendoring: no import path back into architecture/ or data/."""
    for name in ("kwta.py", "kwta_v2.py", "randk.py"):
        src = _read(_VENDORED, name)
        assert "architecture.modules" not in src, name
        assert "import data" not in src and "from data" not in src, name


def test_version_is_a_nonempty_string():
    assert isinstance(scarce.__version__, str) and scarce.__version__
