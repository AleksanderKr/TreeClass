"""
Level 3 validation: TreeClass on public UCI datasets, compared with chefboost.

Datasets (put the original UCI files in data/):
  iris.data      - Iris: 150 cases, 4 numeric attributes, 3 classes
  mushroom.data  - Mushroom: 8124 cases, 22 categorical attributes,
                   "?" = missing value (attribute stalk-root)

Every method is evaluated with the same stratified k-fold split.
Reported per fold and averaged: accuracy (and std), number of leaves,
time of training + prediction.

Run from the repository root:
    python benchmarks/benchmark.py              # 10 folds, TreeClass + chefboost
    python benchmarks/benchmark.py --folds 3 --no-chefboost
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import c45  # noqa: E402
import id3  # noqa: E402
import visualize as viz  # noqa: E402

DATA = ROOT / "data"
TARGET = "class"

MUSHROOM_COLUMNS = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape",
    "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
    "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
    "ring-number", "ring-type", "spore-print-color", "population", "habitat",
]
IRIS_COLUMNS = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_iris():
    return pd.read_csv(DATA / "iris.data", header=None, names=IRIS_COLUMNS).dropna()


def load_mushroom(missing_as_nan=True):
    """missing_as_nan=False keeps "?" as an ordinary category."""
    na = ["?"] if missing_as_nan else []
    return pd.read_csv(DATA / "mushroom.data", header=None, names=MUSHROOM_COLUMNS,
                       na_values=na, keep_default_na=False)


def stratified_folds(labels, k, seed=0):
    """Fold number for every row; classes are spread evenly over the folds."""
    rng = np.random.default_rng(seed)
    folds = np.empty(len(labels), dtype=int)
    for cls in pd.unique(labels):
        idx = np.flatnonzero(labels.to_numpy() == cls)
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % k
    return folds


# ---------------------------------------------------------------------------
# Methods: each takes (train, test, features) and returns (predictions, leaves)
# ---------------------------------------------------------------------------

def treeclass_id3(train, test, features):
    tree = id3.fit(train, TARGET, features)
    return id3.predict(tree, test), viz.count_leaves(tree)


def treeclass_c45(prune):
    def run(train, test, features):
        raw, pruned = c45.fit(train, TARGET, features)
        tree = pruned if prune else raw
        return c45.predict(tree, test), viz.count_leaves(tree)
    return run


def chefboost_method(algorithm):
    """chefboost expects the target as the last column named "Decision" and
    text columns of dtype object (it does not recognise the pandas 3 'str' dtype)."""
    def run(train, test, features):
        from chefboost import Chefboost as chef
        df = train[features + [TARGET]].rename(columns={TARGET: "Decision"})
        df = df.astype({c: object for c in df.columns
                        if not pd.api.types.is_numeric_dtype(df[c])})
        model = chef.fit(df, config={"algorithm": algorithm, "enableParallelism": False},
                         target_label="Decision")
        rows = test[features].astype(object).to_numpy().tolist()
        return [chef.predict(model, param=row) for row in rows], None
    return run


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(name, data, method, k, seed):
    features = [c for c in data.columns if c != TARGET]
    folds = stratified_folds(data[TARGET], k, seed)
    accs, leaves, times = [], [], []
    for f in range(k):
        train, test = data[folds != f], data[folds == f]
        start = time.perf_counter()
        predictions, n_leaves = method(train, test, features)
        times.append(time.perf_counter() - start)
        accs.append(float(np.mean(np.asarray(predictions) == test[TARGET].to_numpy())))
        leaves.append(n_leaves)
    return {
        "method": name,
        "accuracy": np.mean(accs),
        "accuracy_std": np.std(accs),
        "leaves": np.mean(leaves) if None not in leaves else np.nan,
        "time_s": np.mean(times),
    }


def methods(use_chefboost):
    result = [
        ("TreeClass ID3", treeclass_id3, "categorical"),
        ("TreeClass C4.5 (unpruned)", treeclass_c45(prune=False), "nan"),
        ("TreeClass C4.5 (pruned)", treeclass_c45(prune=True), "nan"),
    ]
    if use_chefboost:
        result += [("chefboost ID3", chefboost_method("ID3"), "categorical"),
                   ("chefboost C4.5", chefboost_method("C4.5"), "categorical")]
    return result


def run_benchmark(datasets, k=10, seed=0, use_chefboost=True):
    rows = []
    for dataset in datasets:
        for name, method, missing in methods(use_chefboost):
            if dataset == "iris":
                if name.endswith("ID3"):          # ID3 has no numeric splits
                    continue
                data = load_iris()
            else:
                # "?" is a missing value for TreeClass C4.5; ID3 and chefboost
                # get it as an ordinary category
                data = load_mushroom(missing_as_nan=(missing == "nan"))
            try:
                row = evaluate(name, data, method, k, seed)
            except Exception as err:              # keep the table even if one tool fails
                row = {"method": name, "error": f"{type(err).__name__}: {err}"}
            rows.append({"dataset": dataset, **row})
            print(rows[-1])
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="TreeClass benchmark on UCI datasets")
    ap.add_argument("--datasets", nargs="+", default=["iris", "mushroom"])
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-chefboost", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "output" / "benchmark.csv"))
    args = ap.parse_args(argv)

    table = run_benchmark(args.datasets, args.folds, args.seed, not args.no_chefboost)
    Path(args.out).parent.mkdir(exist_ok=True)
    table.to_csv(args.out, index=False)
    print()
    print(table.round(4).to_string(index=False))
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()