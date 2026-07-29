"""Vendored copies of the mechanism modules the library needs at runtime.

These are BYTE-IDENTICAL copies of architecture/modules/{kwta,kwta_v2,randk}.py
(kwta_v2 differs only in its one internal import path). The research tree stays
the frozen source of record; shipping copies here means installing `scarce` adds
only `scarce` to a user environment, never top-level `architecture`/`data`.
tests/test_packaging.py guards the copies against drift.
"""
