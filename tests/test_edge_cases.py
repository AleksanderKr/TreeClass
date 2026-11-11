"""
Edge cases and properties raised during internal review.

Run from the repository root:  python -m pytest
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import c45
import id3
import visualize as viz

DATA = Path(__file__).resolve().parent.parent / "data"
CATEGORICAL = pd.read_csv(DATA / "play_tennis.csv")
NUMERIC = pd.read_csv(DATA / "play_tennis_numeric.csv")
FEATURES = ["Outlook", "Temperature", "Humidity", "Wind"]


def close(a, b, tol=1e-9):
    return abs(a - b) < tol


# --- entropy -----------------------------------------------------------------

def test_entropy_of_pure_and_empty_sets_is_exactly_zero():
    assert id3.entropy(pd.Series(["Yes"] * 4)) == 0.0
    assert id3.entropy(pd.Series([], dtype=object)) == 0.0
    assert c45.entropy([4, 0]) == 0.0
    assert c45.entropy([]) == 0.0


# --- degenerate splits -------------------------------------------------------

def test_constant_attribute_is_not_used():
    df = pd.DataFrame({"a": ["x"] * 6, "b": [1, 2, 3, 4, 5, 6], "y": list("ppnnpn")})
    raw, _ = c45.fit(df, "y", ["a", "b"])
    assert raw["feature"] == "b"


def test_no_informative_attribute_gives_a_leaf():
    df = pd.DataFrame({"a": list("xxyy"), "y": list("pnpn")})   # gain = 0
    raw, _ = c45.fit(df, "y", ["a"])
    assert raw["type"] == "leaf"


def test_empty_branch_gets_parent_majority():
    """A branch reached by no training case becomes a leaf with the parent's class."""
    tree = id3.fit(NUMERIC, "Play", FEATURES)       # numbers treated as categories
    empty = [(node, child) for node in _all_nodes(tree)
             for child in node.get("branches", {}).values()
             if child["type"] == "leaf" and not child["counts"]]
    assert empty                                    # this dataset produces such branches
    for parent, leaf in empty:
        assert leaf["label"] == parent["label"]


def _all_nodes(tree):
    yield tree
    for child in tree.get("branches", {}).values():
        yield from _all_nodes(child)


# --- review e-mail, point 3: pruning scope -----------------------------------

def test_pruning_collapses_a_node_whose_children_are_subtrees():
    df = pd.DataFrame({"a": list("yyyxxxxxxyyy"),
                       "b": [2, 2, 3, 2, 2, 2, 2, 3, 1, 3, 2, 0],
                       "y": list("npppnppppppp")})
    raw, pruned = c45.fit(df, "y", ["a", "b"])
    assert viz.depth(raw) == 3
    assert pruned["type"] == "leaf" and pruned["label"] == "p"


# --- review e-mail, point 4: pruning must not modify the unpruned tree -------

def test_pruning_does_not_modify_input_tree():
    rng = np.random.default_rng(0)
    noisy = NUMERIC.copy()
    noisy["Play"] = rng.permutation(noisy["Play"].values)
    raw, _ = c45.fit(noisy, "Play", FEATURES)
    before = copy.deepcopy(raw)
    data = noisy.copy()
    data[c45.W] = 1.0
    pruned = c45.prune(raw, data, "Play")
    assert raw == before
    assert viz.tree_stats(pruned)["leaves"] <= viz.tree_stats(raw)["leaves"]


# --- review e-mail, point 5: identifier columns ------------------------------

def test_identifier_column_is_chosen_unless_excluded():
    tree = id3.fit(CATEGORICAL, "Play", ["Day", *FEATURES])
    assert tree["feature"] == "Day"          # hence the --exclude option


# --- pessimistic error estimate ----------------------------------------------

def test_pessimistic_errors_for_zero_errors_is_exact_binomial_bound():
    for n in (1, 2, 6, 14):
        u = c45.pessimistic_errors(n, 0) / n
        assert close((1 - u) ** n, 0.25)       # P(0 errors | n, U) = CF
    assert close(c45.pessimistic_errors(1, 0), 0.75)


def test_pessimistic_errors_monotonicity():
    assert c45.pessimistic_errors(14, 5) > c45.pessimistic_errors(14, 4) > 4
    assert c45.pessimistic_errors(14, 5, cf=0.1) > c45.pessimistic_errors(14, 5, cf=0.25)
    assert 0.75 < c45.pessimistic_errors(2.4, 0.4) < c45.pessimistic_errors(2.4, 1)
    assert c45.pessimistic_errors(0, 0) == 0.0


# --- missing values and labels -----------------------------------------------

def test_id3_rejects_missing_values():
    data = CATEGORICAL.copy()
    data.loc[0, "Outlook"] = np.nan
    with pytest.raises(ValueError):
        id3.fit(data, "Play", FEATURES)


def test_c45_rejects_missing_labels():
    data = NUMERIC.copy()
    data.loc[0, "Play"] = np.nan
    with pytest.raises(ValueError):
        c45.fit(data, "Play", FEATURES)


def test_unseen_categorical_value_uses_node_majority():
    tree = id3.fit(CATEGORICAL, "Play", FEATURES)
    row = pd.DataFrame([{"Outlook": "Snow", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak"}])
    assert id3.predict(tree, row) == [tree["label"]]
    raw, _ = c45.fit(NUMERIC, "Play", FEATURES)
    row = pd.DataFrame([{"Outlook": "Snow", "Temperature": 70, "Humidity": 70, "Wind": "Weak"}])
    assert c45.predict(raw, row) == [raw["label"]]