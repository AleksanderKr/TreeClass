"""
Level 3 validation: TreeClass on public UCI datasets, compared with chefboost
and with Weka J48 and Id3 (run through Java, see benchmarks/j48_validation.py).

Datasets (put the original UCI files in data/):
  iris.data           - Iris: 150 cases, 4 numeric attributes, 3 classes
  mushroom.data       - Mushroom: 8124 cases, 22 categorical attributes,
                        "?" = missing value (attribute stalk-root)
  house-votes-84.data - Congressional Voting Records: 435 cases, 16 categorical
                        attributes (y/n), 2 classes, "?" = missing value

Every method is evaluated with the same stratified k-fold split.
Reported as the mean over the folds: accuracy (and its std), macro-F1,
tree size (nodes, leaves, depth), time in milliseconds of building the tree
(fit_ms) and of classifying the test cases (predict_ms). For C4.5, fit_ms
includes pruning in both variants, because c45.fit always returns the unpruned
and the pruned tree. The saved table is rounded to 4 decimals (accuracy,
macro-F1), 2 decimals (times) and 1 decimal (tree size).

Run from the repository root:
    python benchmarks/benchmark.py
    python benchmarks/benchmark.py --folds 3 --no-chefboost --no-weka
    python benchmarks/benchmark.py --datasets mushroom --fractions 0.1 0.25 0.5 1 \
        --out output/benchmark_scaling.csv
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import c45
import id3
import visualize as viz

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
VOTES_COLUMNS = [
    "class", "handicapped-infants", "water-project-cost-sharing",
    "adoption-of-the-budget-resolution", "physician-fee-freeze", "el-salvador-aid",
    "religious-groups-in-schools", "anti-satellite-test-ban", "aid-to-nicaraguan-contras",
    "mx-missile", "immigration", "synfuels-corporation-cutback", "education-spending",
    "superfund-right-to-sue", "crime", "duty-free-exports",
    "export-administration-act-south-africa",
]



def load_iris():
    return pd.read_csv(DATA / "iris.data", header=None, names=IRIS_COLUMNS).dropna()


def load_mushroom(missing_as_nan=True):
    """missing_as_nan=False keeps "?" as an ordinary category."""
    na = ["?"] if missing_as_nan else []
    return pd.read_csv(DATA / "mushroom.data", header=None, names=MUSHROOM_COLUMNS,
                       na_values=na, keep_default_na=False)


def load_votes(missing_as_nan=True):
    """missing_as_nan=False keeps "?" as an ordinary category."""
    na = ["?"] if missing_as_nan else []
    return pd.read_csv(DATA / "house-votes-84.data", header=None, names=VOTES_COLUMNS,
                       na_values=na, keep_default_na=False)


LOADERS = {"mushroom": load_mushroom, "votes": load_votes}


def stratified_folds(labels, k, seed=0):
    """Fold number for every row; classes are spread evenly over the folds."""
    rng = np.random.default_rng(seed)
    folds = np.empty(len(labels), dtype=int)
    for cls in pd.unique(labels):
        idx = np.flatnonzero(labels.to_numpy() == cls)
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % k
    return folds



def tree_size(tree):
    stats = viz.tree_stats(tree)
    return {"nodes": stats["internal"] + stats["leaves"], "leaves": stats["leaves"],
            "depth": stats["depth"]}


def timed(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - start


def treeclass_id3(train, test, features):
    """Every method takes (train, test, features) and returns a dict with the
    predictions, fit_s, predict_s and the tree size (NaN when unknown)."""
    tree, fit_s = timed(id3.fit, train, TARGET, features)
    predictions, predict_s = timed(id3.predict, tree, test)
    return {"predictions": predictions, "fit_s": fit_s, "predict_s": predict_s, **tree_size(tree)}


def treeclass_c45(prune):
    def run(train, test, features):
        (raw, pruned), fit_s = timed(c45.fit, train, TARGET, features)
        tree = pruned if prune else raw
        predictions, predict_s = timed(c45.predict, tree, test)
        return {"predictions": predictions, "fit_s": fit_s, "predict_s": predict_s, **tree_size(tree)}
    return run


def chefboost_method(algorithm):
    """chefboost (pip package) as a Python point of comparison. Its C4.5
    differs from Quinlan's (and from TreeClass and J48), as its source shows:
      * the test with the best gain ratio is chosen, without the rule that
        its gain must be at least the average gain;
      * a numeric attribute with more than 20 distinct values is tried only at
        about 10 thresholds (min, max, mean and mean + k*std for k = -3..3);
      * missing values are not handled ("?" is passed as an ordinary category);
      * there is no error-based pruning.
    The first point explains its root Temperature <= 83 on Play Tennis, the
    same split as TreeClass's Temperature <= 84 before the average-gain rule.

    chefboost expects the target as the last column named "Decision" and
    text columns of dtype object (it does not recognise the pandas 3 'str' dtype).
    It writes the tree to outputs/rules/rules.py in the working directory and
    imports it as a module, so that directory must be importable, no stale .pyc
    may be left between folds and the module must be re-imported for every fold.
    The tree size is not available."""
    def run(train, test, features):
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
        model, fit_s = timed(chef.fit, df, config={"algorithm": algorithm, "enableParallelism": False},
                             target_label="Decision")
        rows = test[features].astype(object).to_numpy().tolist()
        predictions, predict_s = timed(lambda: [chef.predict(model, param=row) for row in rows])
        return {"predictions": predictions, "fit_s": fit_s, "predict_s": predict_s,
                "nodes": np.nan, "leaves": np.nan, "depth": np.nan}
    return run


def weka_method(classifier, options, tree_of):
    """Weka classifier run through Java. fit_s is the time printed by Weka
    itself (in steps of 0.01 s), so the start of the Java virtual machine is
    not counted; Weka prints no time of classifying the test cases here, so
    predict_s is NaN. tree_of extracts the tree listing from the model printout."""
    def run(train, test, features):
        import j48_validation as weka
        jar = weka.ensure_weka()
        columns = features + [TARGET]
        header = weka.arff_header(pd.concat([train, test])[columns])
        with tempfile.TemporaryDirectory() as tmp:
            train_file, test_file = Path(tmp) / "train.arff", Path(tmp) / "test.arff"
            weka.write_arff(train_file, header, train[columns])
            weka.write_arff(test_file, header, test[columns])
            predictions, model = weka.run_weka(jar, classifier, train_file, test_file, options)
        return {"predictions": predictions, "fit_s": weka.build_time(model),
                "predict_s": np.nan,
                **weka.tree_text_size(tree_of(model))}
    return run


def macro_f1(truth, predictions):
    """Mean of the per-class F1 scores over the classes of the true labels.
    A prediction outside them (Weka Id3 predicts "?" for a branch reached by
    no training case) counts as an error, but not as an extra class."""
    truth, predictions = np.asarray(truth, dtype=str), np.asarray(predictions, dtype=str)
    scores = []
    for cls in np.unique(truth):
        tp = np.sum((predictions == cls) & (truth == cls))
        fp = np.sum((predictions == cls) & (truth != cls))
        fn = np.sum((predictions != cls) & (truth == cls))
        scores.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores))


def evaluate(name, data, method, k, seed):
    features = [c for c in data.columns if c != TARGET]
    folds = stratified_folds(data[TARGET], k, seed)
    per_fold = []
    for f in range(k):
        train, test = data[folds != f], data[folds == f]
        result = method(train, test, features)
        truth = test[TARGET].astype(str).to_numpy()
        predictions = np.asarray(result.pop("predictions"), dtype=str)
        per_fold.append({"accuracy": np.mean(predictions == truth),
                         "macro_f1": macro_f1(truth, predictions), **result})
    table = pd.DataFrame(per_fold)
    return {"method": name, "accuracy": table["accuracy"].mean(),
            "accuracy_std": table["accuracy"].std(ddof=0),
            **table.drop(columns="accuracy").mean().to_dict()}


def methods(use_chefboost, use_weka):
    result = [
        ("TreeClass ID3", treeclass_id3, "categorical"),
        ("TreeClass C4.5 (unpruned)", treeclass_c45(prune=False), "nan"),
        ("TreeClass C4.5 (pruned)", treeclass_c45(prune=True), "nan"),
    ]
    if use_chefboost:
        result += [("chefboost ID3", chefboost_method("ID3"), "categorical"),
                   ("chefboost C4.5", chefboost_method("C4.5"), "categorical")]
    if use_weka:
        import j48_validation as weka
        j48 = "weka.classifiers.trees.J48"
        result += [("Weka Id3", weka_method("weka.classifiers.trees.Id3", [], weka.id3_tree),
                    "categorical"),
                   ("Weka J48 (unpruned)", weka_method(j48, weka.VARIANTS["unpruned"][0],
                                                       weka.j48_tree), "nan"),
                   ("Weka J48 (pruned)", weka_method(j48, weka.MATCHED, weka.j48_tree), "nan")]
    return result


def sample(data, fraction, seed):
    """Stratified random sample: the same share of every class. The rows
    chosen depend only on the class column, so every method gets the same
    sample, whichever way it reads the missing values."""
    if fraction >= 1:
        return data
    return data.groupby(TARGET, group_keys=False).sample(frac=fraction, random_state=seed)


def run_benchmark(datasets, k=10, seed=0, use_chefboost=True, use_weka=True, fractions=(1.0,)):
    """ID3 is skipped on Iris (no numeric splits). "?" is a missing value for
    C4.5 of TreeClass and Weka; ID3 and chefboost get it as an ordinary category.
    With several fractions every dataset is also run on stratified samples of
    that size, to show how quality and time change with the number of cases.
    A tool that fails gets an error row, so the table is kept."""
    jobs = [(dataset, fraction, entry) for dataset in datasets for fraction in fractions
            for entry in methods(use_chefboost, use_weka)
            if not (dataset == "iris" and "ID3" in entry[0].upper())]
    rows = []
    for i, (dataset, fraction, (name, method, missing)) in enumerate(jobs, 1):
        if dataset == "iris":
            data = load_iris()
        else:
            data = LOADERS[dataset](missing_as_nan=(missing == "nan"))
        data = sample(data, fraction, seed)
        print(f"[{i}/{len(jobs)}] {dataset}, {len(data)} cases, {name} ...", flush=True)
        try:
            row = evaluate(name, data, method, k, seed)
        except Exception as err:
            row = {"method": name, "error": f"{type(err).__name__}: {err}"}
            print(f"    {row['error']}", flush=True)
        rows.append({"dataset": dataset, "fraction": fraction, "cases": len(data), **row})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="TreeClass benchmark on UCI datasets")
    ap.add_argument("--datasets", nargs="+", default=["iris", "mushroom", "votes"])
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-chefboost", action="store_true")
    ap.add_argument("--no-weka", action="store_true", help="skip J48 and Id3 (no Java needed)")
    ap.add_argument("--fractions", nargs="+", type=float, default=[1.0],
                    help="sizes of the stratified samples, e.g. 0.1 0.25 0.5 1")
    ap.add_argument("--out", default=str(ROOT / "output" / "benchmark.csv"))
    args = ap.parse_args(argv)

    table = run_benchmark(args.datasets, args.folds, args.seed,
                          not args.no_chefboost, not args.no_weka, args.fractions)
    for col in ("fit", "predict"):
        table[f"{col}_ms"] = table.pop(f"{col}_s") * 1000
    table = table.round({"accuracy": 4, "accuracy_std": 4, "macro_f1": 4,
                         "nodes": 1, "leaves": 1, "depth": 1, "fit_ms": 2, "predict_ms": 2})
    Path(args.out).parent.mkdir(exist_ok=True)
    table.to_csv(args.out, index=False)
    print()
    print(table.to_string(index=False))
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()