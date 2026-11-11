"""
Level 3 validation (light version for pytest): TreeClass on the UCI datasets.
Skipped when the data files are missing; the chefboost check is skipped when
chefboost is not installed. Full table: python benchmarks/benchmark.py
"""

import pytest

import benchmark as bm

pytestmark = pytest.mark.skipif(
    not (bm.DATA / "iris.data").exists() or not (bm.DATA / "mushroom.data").exists(),
    reason="UCI data files not found in data/",
)


def test_c45_on_iris():
    result = bm.evaluate("C4.5", bm.load_iris(), bm.treeclass_c45(prune=True), k=5, seed=0)
    assert result["accuracy"] >= 0.90


def test_c45_on_mushroom_with_missing_values():
    data = bm.load_mushroom(missing_as_nan=True)
    assert data["stalk-root"].isna().any()
    result = bm.evaluate("C4.5", data, bm.treeclass_c45(prune=True), k=3, seed=0)
    assert result["accuracy"] >= 0.95


def test_chefboost_runs_on_iris():
    pytest.importorskip("chefboost")
    result = bm.evaluate("chefboost", bm.load_iris(), bm.chefboost_method("C4.5"), k=5, seed=0)
    assert result["accuracy"] >= 0.85