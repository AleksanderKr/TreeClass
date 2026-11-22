"""
Validation of the implementation against hand-calculated reference values:
  * the ID3 and C4.5 examples of the article (Play Tennis data),
  * Quinlan (1993): gain and split info with one unknown Outlook value.
Edge cases are tested in test_edge_cases.py.

Run:  python tests/test_article_examples.py   (or: python -m pytest tests/test_article_examples.py)
Run directly, the script also prints the intermediate steps of both
examples in the order used in the article.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))   # for direct runs

import c45
import id3
import visualize as viz


COLUMNS = ["Day", "Outlook", "Temperature", "Humidity", "Wind", "Play"]
CATEGORICAL = pd.DataFrame([
    [1, "Sunny", "Hot", "High", "Weak", "No"],
    [2, "Sunny", "Hot", "High", "Strong", "No"],
    [3, "Overcast", "Hot", "High", "Weak", "Yes"],
    [4, "Rain", "Mild", "High", "Weak", "Yes"],
    [5, "Rain", "Cool", "Normal", "Weak", "Yes"],
    [6, "Rain", "Cool", "Normal", "Strong", "No"],
    [7, "Overcast", "Cool", "Normal", "Strong", "Yes"],
    [8, "Sunny", "Mild", "High", "Weak", "No"],
    [9, "Sunny", "Cool", "Normal", "Weak", "Yes"],
    [10, "Rain", "Mild", "Normal", "Weak", "Yes"],
    [11, "Sunny", "Mild", "Normal", "Strong", "Yes"],
    [12, "Overcast", "Mild", "High", "Strong", "Yes"],
    [13, "Overcast", "Hot", "Normal", "Weak", "Yes"],
    [14, "Rain", "Mild", "High", "Strong", "No"],
], columns=COLUMNS)
NUMERIC = pd.DataFrame([
    [1, "Sunny", 85, 85, "Weak", "No"],
    [2, "Sunny", 80, 90, "Strong", "No"],
    [3, "Overcast", 83, 78, "Weak", "Yes"],
    [4, "Rain", 70, 96, "Weak", "Yes"],
    [5, "Rain", 68, 80, "Weak", "Yes"],
    [6, "Rain", 65, 70, "Strong", "No"],
    [7, "Overcast", 64, 65, "Strong", "Yes"],
    [8, "Sunny", 72, 95, "Weak", "No"],
    [9, "Sunny", 69, 70, "Weak", "Yes"],
    [10, "Rain", 75, 80, "Weak", "Yes"],
    [11, "Sunny", 75, 70, "Strong", "Yes"],
    [12, "Overcast", 72, 90, "Strong", "Yes"],
    [13, "Overcast", 81, 75, "Weak", "Yes"],
    [14, "Rain", 71, 80, "Strong", "No"],
], columns=COLUMNS)
FEATURES = ["Outlook", "Temperature", "Humidity", "Wind"]


def close(a, b, tol=5e-5):
    return abs(a - b) < tol


def c45_tests(data):
    """C4.5 candidate tests on a node holding `data`, keyed by attribute."""
    data = data.copy()
    data[c45.W] = 1.0
    numeric = {f for f in FEATURES if c45.is_numeric(data[f])}
    domains = {f: sorted(NUMERIC[f].unique().tolist(), key=str) for f in FEATURES if f not in numeric}
    tests = c45.candidate_tests(data, "Play", FEATURES, domains, numeric, min_cases=1)
    return {t["feature"]: t for t in tests}, c45.choose_test(tests)


# --- entropy of the whole dataset -------------------------------------------

def test_entropy_of_whole_dataset():
    assert close(id3.entropy(CATEGORICAL["Play"]), 0.9403)
    assert close(c45.entropy([9, 5]), 0.9403)


# --- ID3 example of the article ---------------------------------------------

def test_id3_gains():
    d = CATEGORICAL
    expected = {"Outlook": 0.2467, "Humidity": 0.1518, "Wind": 0.0481, "Temperature": 0.0292}
    for f, g in expected.items():
        assert close(id3.information_gain(d, f, "Play"), g, 1e-4), f
    sunny, rain = d[d.Outlook == "Sunny"], d[d.Outlook == "Rain"]
    for f, g in {"Humidity": 0.9710, "Temperature": 0.5710, "Wind": 0.0200}.items():
        assert close(id3.information_gain(sunny, f, "Play"), g, 1e-4), f
    for f, g in {"Wind": 0.9710, "Temperature": 0.0200, "Humidity": 0.0200}.items():
        assert close(id3.information_gain(rain, f, "Play"), g, 1e-4), f


def test_id3_tree():
    tree = id3.fit(CATEGORICAL, "Play", FEATURES)
    assert tree["feature"] == "Outlook"
    assert tree["branches"]["Sunny"]["feature"] == "Humidity"
    assert tree["branches"]["Rain"]["feature"] == "Wind"
    assert tree["branches"]["Overcast"]["label"] == "Yes"
    assert viz.tree_stats(tree) == {"internal": 3, "leaves": 5, "depth": 2}
    assert id3.predict(tree, CATEGORICAL) == CATEGORICAL["Play"].tolist()


# --- C4.5 example of the article (Release 7) --------------------------------

def test_c45_root_table():
    tests, chosen = c45_tests(NUMERIC)
    expected = {   # feature: (threshold, gain, split info, gain ratio) - article table
        "Outlook": (None, 0.2467, 1.5774, 0.1564),
        "Temperature": (84.0, 0.1134, 0.3712, 0.3055),
        "Humidity": (67.5, 0.0477, 0.3712, 0.1285),   # tie with 95.5 -> smaller
        "Wind": (None, 0.0481, 0.9852, 0.0488),
    }
    for f, (t, g, si, gr) in expected.items():
        test = tests[f]
        assert test["threshold"] == t, f
        assert close(test["gain"], g, 1e-4) and close(test["split_info"], si, 1e-4)
        assert close(test["gain_ratio"], gr, 1e-4), f
    assert close(np.mean([t["gain"] for t in tests.values()]), 0.1140, 1e-4)
    assert chosen["feature"] == "Outlook"     # Temperature has the best ratio but gain < average


def test_c45_sunny_and_rain_nodes():
    tests, chosen = c45_tests(NUMERIC[NUMERIC.Outlook == "Sunny"])
    assert chosen["feature"] == "Humidity" and chosen["threshold"] == 77.5
    assert close(chosen["gain_ratio"], 1.0)
    assert tests["Temperature"]["threshold"] == 70.5
    assert close(tests["Temperature"]["gain"], 0.3219, 1e-4)
    assert close(tests["Temperature"]["gain_ratio"], 0.4459, 1e-4)
    assert close(tests["Wind"]["gain"], 0.0200, 1e-4) and close(tests["Wind"]["gain_ratio"], 0.0206, 1e-4)
    gains = [t["gain"] for f, t in tests.items() if f != "Outlook"]
    assert close(np.mean(gains), 0.4376, 1e-4)

    tests, chosen = c45_tests(NUMERIC[NUMERIC.Outlook == "Rain"])
    assert chosen["feature"] == "Wind" and close(chosen["gain_ratio"], 1.0)
    assert tests["Temperature"]["threshold"] == 66.5 and tests["Humidity"]["threshold"] == 75.0
    for f in ("Temperature", "Humidity"):
        assert close(tests[f]["gain"], 0.3219, 1e-4) and close(tests[f]["gain_ratio"], 0.4459, 1e-4)
    gains = [t["gain"] for f, t in tests.items() if f != "Outlook"]
    assert close(np.mean(gains), 0.5383, 1e-4)


def test_c45_tree_and_pruning():
    raw, pruned = c45.fit(NUMERIC, "Play", FEATURES)
    assert raw["feature"] == "Outlook"
    sunny = raw["branches"]["Sunny"]
    assert sunny["feature"] == "Humidity" and sunny["threshold"] == 77.5
    assert raw["branches"]["Rain"]["feature"] == "Wind"
    assert viz.tree_stats(raw) == {"internal": 3, "leaves": 5, "depth": 2}
    assert c45.predict(raw, NUMERIC) == NUMERIC["Play"].tolist()
    assert viz.tree_to_text(pruned) == viz.tree_to_text(raw)    # pruning keeps this tree


# --- missing values ----------------------------------------------------------

def test_missing_value_quinlan_1993():
    """Quinlan (1993): Outlook of one case unknown -> gain 0.199, split info 1.809."""
    data = NUMERIC.copy()
    data.loc[data.Day == 12, "Outlook"] = np.nan
    tests, _ = c45_tests(data)
    assert close(tests["Outlook"]["gain"], 0.199, 5e-4)
    assert close(tests["Outlook"]["split_info"], 1.809, 5e-4)

    # Release 7 behaviour: the lower Outlook gain lowers the average gain to ~0.102,
    # so "Temperature <= 84" (gain 0.1134, best gain ratio) becomes eligible and wins.
    raw, _ = c45.fit(data, "Play", FEATURES)
    assert raw["feature"] == "Temperature" and raw["threshold"] == 84.0

    # Fractional weights: without Temperature the root is Outlook and the unknown
    # case is split 5/13, 3/13, 5/13 between Sunny, Overcast and Rain.
    raw, _ = c45.fit(data, "Play", ["Outlook", "Humidity", "Wind"])
    assert raw["feature"] == "Outlook"
    overcast = raw["branches"]["Overcast"]
    assert close(sum(overcast["counts"].values()), 3 + 3 / 13, 1e-4)   # 3 known + 3/13 of the unknown case
    prediction = c45.predict(raw, pd.DataFrame([{"Outlook": np.nan, "Temperature": 70,
                                                 "Humidity": 70, "Wind": "Weak"}]))
    assert prediction == ["Yes"]


# --- intermediate steps, printed when the file is run directly ---------------

def distribution(data):
    """[n+, n-] as in the article: Yes first, then No."""
    return f"[{(data.Play == 'Yes').sum()},{(data.Play == 'No').sum()}]"


def days(data):
    return "{" + ",".join(str(d) for d in data.Day) + "}"


def label(test):
    if test["threshold"] is None:
        return test["feature"]
    return f"{test['feature']} <= {test['threshold']:g}"


def print_id3_node(title, data, features):
    print(f"\n{title}: Days {days(data)}, {distribution(data)}, "
          f"Entropy = {id3.entropy(data.Play):.4f}")
    gains = {f: id3.information_gain(data, f, "Play") for f in features}
    for f, g in sorted(gains.items(), key=lambda item: -item[1]):
        print(f"  Gain({f}) = {g:.4f}")


def print_id3_example():
    print("=" * 70 + "\nID3: Play Tennis (categorical)\n" + "=" * 70)
    d = CATEGORICAL
    print_id3_node("S", d, FEATURES)
    print("\nSplit on Outlook:")
    for value in ("Sunny", "Overcast", "Rain"):
        subset = d[d.Outlook == value]
        print(f"  {value}: Days {days(subset)}, {distribution(subset)}, "
              f"Entropy = {id3.entropy(subset.Play):.4f}")
    remaining = [f for f in FEATURES if f != "Outlook"]
    print_id3_node("Sunny", d[d.Outlook == "Sunny"], remaining)
    print_id3_node("Rain", d[d.Outlook == "Rain"], remaining)
    tree = id3.fit(d, "Play", FEATURES)
    print(f"\nResulting tree {viz.tree_stats(tree)}:")
    viz.print_tree(tree)


def print_c45_node(title, data):
    tests, chosen = c45_tests(data)
    print(f"\n{title}: Days {days(data)}, {distribution(data)}, "
          f"Entropy = {c45.entropy([(data.Play == 'Yes').sum(), (data.Play == 'No').sum()]):.4f}")
    print(f"  {'Test':<22}{'Gain':>8}{'SplitInfo':>11}{'GainRatio':>11}")
    for f in FEATURES:
        if f in tests:
            t = tests[f]
            print(f"  {label(t):<22}{t['gain']:>8.4f}{t['split_info']:>11.4f}{t['gain_ratio']:>11.4f}")
    average = np.mean([t["gain"] for t in tests.values()])
    eligible = [label(t) for t in tests.values() if t["gain"] >= average - 1e-12]
    print(f"  Average gain = {average:.4f}; tests with gain >= average: {', '.join(eligible)}")
    print(f"  Selected: {label(chosen)}")


def print_c45_example():
    print("\n" + "=" * 70 + "\nC4.5: Play Tennis (numeric Temperature and Humidity)\n" + "=" * 70)
    d = NUMERIC
    for f in ("Temperature", "Humidity"):
        values = np.sort(d[f].unique())
        thresholds = (values[:-1] + values[1:]) / 2
        print(f"\n{f}: values {', '.join(f'{v:g}' for v in values)}")
        print(f"  candidate thresholds {', '.join(f'{t:g}' for t in thresholds)}")
    print_c45_node("S", d)
    print_c45_node("Sunny", d[d.Outlook == "Sunny"])
    print_c45_node("Rain", d[d.Outlook == "Rain"])
    raw, pruned = c45.fit(d, "Play", FEATURES)
    print(f"\nTree before pruning {viz.tree_stats(raw)}:")
    viz.print_tree(raw)
    same = viz.tree_to_text(pruned) == viz.tree_to_text(raw)
    print(f"Pruning (CF = 25%) {'keeps this tree' if same else 'changes this tree'}.\n")


if __name__ == "__main__":
    print_id3_example()
    print_c45_example()
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, func in tests:
        try:
            func()
            print(f"PASS  {name}")
        except AssertionError as err:
            failed += 1
            print(f"FAIL  {name}  {err}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)