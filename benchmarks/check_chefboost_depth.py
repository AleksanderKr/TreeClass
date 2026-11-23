"""
Diagnostic script (not part of TreeClass): why does the accuracy of chefboost
ID3 on Votes (0.9358) differ from TreeClass ID3 (0.9289) and Weka Id3?

Hypothesis: chefboost limits the depth of every tree to max_depth = 5 by
default (commons/functions.py, initializeParams), also for ID3 and C4.5
without GBM; at that depth createBranch (training/Training.py) returns the
majority class instead of splitting further. TreeClass ID3 trees on Votes
have depth 7.6 on average, the Mushroom trees only 4, which would explain why
only Votes differs.

Check: the same 10 folds (seed 0) as benchmarks/benchmark.py, chefboost run
with its default configuration and with max_depth = 1000 (no limit).
If the hypothesis holds, chefboost ID3 without the limit gives the TreeClass
accuracy (0.9289) and the same prediction for every test case.

Put this file in benchmarks/ and run from the repository root:
    python benchmarks/check_chefboost_depth.py
The table is printed and saved to output/chefboost_depth_check.csv.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as bm  # also puts src/ on sys.path

import c45
import id3
import visualize as viz

TARGET = bm.TARGET
NO_LIMIT = 1000


def rules_depth(path):
    """Depth (edges on the longest root-to-leaf path) of the tree that
    chefboost wrote to outputs/rules/rules.py, read from the indentation:
    a return statement is a leaf, its depth is the number of if-levels above it."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    indent = lambda line: len(line) - len(line.lstrip())
    if_levels = sorted({indent(l) for l in lines if re.match(r"\s*(if|elif) obj\[", l)})
    returns = [indent(l) for l in lines if l.strip().startswith("return")]
    return max(sum(level < r for level in if_levels) for r in returns)


def chefboost_run(train, test, features, algorithm, max_depth):
    """As chefboost_method in benchmark.py, with max_depth set explicitly
    (None = chefboost default). Returns the predictions and the tree depth."""
    from chefboost import Chefboost as chef
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    sys.dont_write_bytecode = True
    for module in [m for m in sys.modules if m == "outputs" or m.startswith("outputs.")]:
        del sys.modules[module]

    df = train[features + [TARGET]].rename(columns={TARGET: "Decision"})
    df = df.astype({c: object for c in df.columns
                    if not pd.api.types.is_numeric_dtype(df[c])})
    config = {"algorithm": algorithm, "enableParallelism": False}
    if max_depth is not None:
        config["max_depth"] = max_depth
    model = chef.fit(df, config=config, target_label="Decision")
    depth = rules_depth(Path.cwd() / "outputs" / "rules" / "rules.py")
    rows = test[features].astype(object).to_numpy().tolist()
    predictions = [chef.predict(model, param=row) for row in rows]
    return np.asarray(predictions, dtype=str), depth


def treeclass_run(train, test, features, algorithm):
    if algorithm == "ID3":
        tree = id3.fit(train, TARGET, features)
        predictions = id3.predict(tree, test)
    else:
        _, tree = c45.fit(train, TARGET, features)  # pruned tree
        predictions = c45.predict(tree, test)
    return np.asarray(predictions, dtype=str), viz.tree_stats(tree)["depth"]


def load(dataset, missing_as_nan):
    if dataset == "iris":
        return bm.load_iris()
    return bm.LOADERS[dataset](missing_as_nan=missing_as_nan)


def check(dataset, algorithm, k=10, seed=0):
    """chefboost (default, no limit) and TreeClass on the same folds.
    chefboost always gets "?" as a category; TreeClass ID3 too, TreeClass C4.5
    gets it as a missing value (as in benchmark.py)."""
    data_cat = load(dataset, missing_as_nan=False)
    data_tc = data_cat if algorithm == "ID3" else load(dataset, missing_as_nan=True)
    features = [c for c in data_cat.columns if c != TARGET]
    folds = bm.stratified_folds(data_cat[TARGET], k, seed)

    runs = {"chefboost default": [], f"chefboost max_depth={NO_LIMIT}": [], "TreeClass": []}
    truth_all = []
    for f in range(k):
        tr, te = folds != f, folds == f
        truth_all.append(data_cat[te][TARGET].astype(str).to_numpy())
        runs["chefboost default"].append(
            chefboost_run(data_cat[tr], data_cat[te], features, algorithm, None))
        runs[f"chefboost max_depth={NO_LIMIT}"].append(
            chefboost_run(data_cat[tr], data_cat[te], features, algorithm, NO_LIMIT))
        runs["TreeClass"].append(treeclass_run(data_tc[tr], data_tc[te], features, algorithm))

    tc_pred = runs["TreeClass"]
    rows = []
    for name, per_fold in runs.items():
        acc = [np.mean(p == t) for (p, _), t in zip(per_fold, truth_all)]
        agree = [np.mean(p == q) for (p, _), (q, _) in zip(per_fold, tc_pred)]
        depths = [d for _, d in per_fold]
        rows.append({"dataset": dataset, "algorithm": algorithm, "run": name,
                     "accuracy": round(float(np.mean(acc)), 4),
                     "agreement_with_TreeClass": round(float(np.mean(agree)), 4),
                     "test_cases_differing_from_TreeClass":
                         int(sum(np.sum(p != q) for (p, _), (q, _) in zip(per_fold, tc_pred))),
                     "mean_depth": round(float(np.mean(depths)), 1),
                     "max_depth": int(np.max(depths)),
                     "folds_with_depth_5": int(np.sum(np.asarray(depths) == 5))})
    return rows


def main():
    import chefboost
    print("chefboost version:", getattr(chefboost, "__version__", "unknown"))
    try:
        from chefboost.commons import functions
        print("default max_depth:", functions.initializeParams({}).get("max_depth"))
    except Exception as err:
        print("initializeParams not readable:", type(err).__name__, err)

    jobs = [("votes", "ID3"), ("mushroom", "ID3"),
            ("votes", "C4.5"), ("iris", "C4.5"), ("mushroom", "C4.5")]
    rows = []
    for dataset, algorithm in jobs:
        print(f"\n=== {dataset}, {algorithm} ===", flush=True)
        rows += check(dataset, algorithm)

    table = pd.DataFrame(rows)
    out = bm.ROOT / "output" / "chefboost_depth_check.csv"
    out.parent.mkdir(exist_ok=True)
    table.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print()
    print(table.to_string(index=False))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()