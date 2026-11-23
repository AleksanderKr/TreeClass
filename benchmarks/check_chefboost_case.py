"""
Diagnostic script (not part of TreeClass): the test cases of Votes on which
chefboost ID3 without the depth limit (max_depth = 1000) and TreeClass ID3
predict different classes (one case in check_chefboost_depth.py).

For every such case the script prints the path of the case through the
TreeClass tree: at each node the attribute, the value of the case, the class
counts and the two best information gains (to show ties), and at the end why
TreeClass predicts its class (leaf, or a value without a training case at the
node). It also saves the chefboost tree of that fold
(output/chefboost_rules_fold<k>.py) for comparison.

Put this file in benchmarks/ and run from the repository root:
    python benchmarks/check_chefboost_case.py
"""

import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as bm  # also puts src/ on sys.path
import check_chefboost_depth as cd

import id3

TARGET = bm.TARGET


def gains(data, features):
    """Information gains of the remaining attributes, best first."""
    g = [(id3.information_gain(data, f, TARGET), f) for f in features]
    return sorted(g, key=lambda x: -x[0])


def trace(tree, train, row, features):
    """Path of one case through the TreeClass tree, with the training rows
    that reach every node."""
    node, data, remaining, depth = tree, train, list(features), 0
    while node["type"] == "node":
        f = node["feature"]
        top = gains(data, remaining)[:2]
        tie = len(top) == 2 and abs(top[0][0] - top[1][0]) < 1e-9
        print(f"  depth {depth}: {f} = {row[f]!r}, counts {node['counts']}, "
              f"majority {node['label']!r}")
        print("      best gains: " + ", ".join(f"{name} {g:.17g}" for g, name in top)
              + ("   <-- tie" if tie else ""))
        if row[f] not in node["branches"]:
            print(f"  value {row[f]!r} not in the tree: majority of this node, {node['label']!r}")
            return
        data = data[data[f] == row[f]]
        remaining = [x for x in remaining if x != f]
        node, depth = node["branches"][row[f]], depth + 1
    reason = "empty branch, majority of the parent node" if not node["counts"] else "leaf"
    counts = node["counts"]
    tie = len(counts) > 1 and len(set(counts.values())) == 1
    print(f"  depth {depth}: {reason}, counts {counts}, label {node['label']!r}"
          + ("   <-- tie of classes" if tie else "")
          + (f", no attributes left" if not remaining else ""))


def main():
    data = bm.load_votes(missing_as_nan=False)
    features = [c for c in data.columns if c != TARGET]
    folds = bm.stratified_folds(data[TARGET], 10, 0)
    out = bm.ROOT / "output"
    out.mkdir(exist_ok=True)

    found = 0
    for f in range(10):
        train, test = data[folds != f], data[folds == f]
        chef_pred, _ = cd.chefboost_run(train, test, features, "ID3", cd.NO_LIMIT)
        tree = id3.fit(train, TARGET, features)
        tc_pred = np.asarray(id3.predict(tree, test), dtype=str)
        differ = np.flatnonzero(chef_pred != tc_pred)
        if differ.size == 0:
            continue
        rules = out / f"chefboost_rules_fold{f}.py"
        shutil.copy(Path.cwd() / "outputs" / "rules" / "rules.py", rules)
        for i in differ:
            found += 1
            row = test.iloc[i]
            print(f"\n=== fold {f}, row {test.index[i]} of house-votes-84.data "
                  f"(0-based), true class {row[TARGET]!r}: "
                  f"TreeClass {tc_pred[i]!r}, chefboost {chef_pred[i]!r} ===")
            print("  case: " + ", ".join(f"{c}={row[c]}" for c in features))
            trace(tree, train, row, features)
        print(f"  chefboost tree of this fold saved: {rules}")
    print(f"\ncases that differ: {found}")


if __name__ == "__main__":
    main()