"""
Level 3 validation against the reference implementations in Weka:
J48 (C4.5) and Id3.

J48 is configured as close as possible to TreeClass:
  -C 0.25   pruning confidence (CF)          -M 1   min. cases per branch
  -S        no subtree raising               -O     do not collapse the tree
  -J        no MDL correction (Release 7)    -doNotMakeSplitPointActualValue
                                                    thresholds at midpoints
Remaining known differences, both for numeric attributes only: J48 chooses
thresholds by gain (Release 8), TreeClass by gain ratio (Release 7), and J48
requires each side of a threshold to hold at least 10% of the average number
of cases per class (at least -M, at most 25). For purely categorical data
(Mushroom, Votes) the two build the same trees.

Id3 (Weka package simpleEducationalLearningSchemes) accepts only categorical
attributes without missing values, so it is compared on Mushroom and Votes
with "?" as an ordinary category. For a branch reached by no training case
Id3 predicts no class, while TreeClass predicts the parent's majority class.

Both tools get the same stratified folds as benchmarks/benchmark.py.
Reported per dataset and variant: accuracy of TreeClass and Weka, the share
of test cases on which both predict the same class (agreement), and leaves.
The trees built on the full data are saved to output/ and compared line by
line (column same_tree); differences in notation only are ignored
(e.g. J48 writes 50.0 where TreeClass writes 50).

Requires Java. The Weka jars are downloaded from Maven Central on first use
(Weka is GPL-licensed, so it is not stored in this repository).

Run from the repository root:
    python benchmarks/j48_validation.py
    python benchmarks/j48_validation.py --folds 3 --datasets mushroom
    python benchmarks/j48_validation.py --download-only      # used by MyBinder
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import benchmark as bm
import visualize as viz

MAVEN = "https://repo1.maven.org/maven2/nz/ac/waikato/cms/weka"
WEKA_VERSION = "3.8.6"
LIB = Path(__file__).resolve().parent / "lib"
JARS = {
    "weka.jar": f"{MAVEN}/weka-stable/{WEKA_VERSION}/weka-stable-{WEKA_VERSION}.jar",
    "bounce.jar": f"{MAVEN}/thirdparty/bounce/0.18/bounce-0.18.jar",
    "simpleEducationalLearningSchemes.jar":
        f"{MAVEN}/simpleEducationalLearningSchemes/1.0.2/simpleEducationalLearningSchemes-1.0.2.jar",
}

MATCHED = ["-C", "0.25", "-M", "1", "-S", "-O", "-J", "-doNotMakeSplitPointActualValue"]
VARIANTS = {
    "pruned": (MATCHED, bm.treeclass_c45(prune=True)),
    "unpruned": (["-U", "-M", "1", "-O", "-J", "-doNotMakeSplitPointActualValue"],
                 bm.treeclass_c45(prune=False)),
}
ID3_DATASETS = ("mushroom", "votes")


def ensure_weka():
    """Download the jars on first use and return the Java classpath.
    weka-stable needs the "bounce" library at run time."""
    LIB.mkdir(parents=True, exist_ok=True)
    for name, url in JARS.items():
        if not (LIB / name).exists():
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, LIB / name)
    return os.pathsep.join(str(LIB / name) for name in JARS)


def _quote(value):
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def arff_header(data):
    """Header with nominal domains taken from the whole dataset, so every
    fold file declares the same values in the same (sorted) order."""
    lines = ["@relation treeclass", ""]
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]) and col != bm.TARGET:
            lines.append(f"@attribute {_quote(col)} numeric")
        else:
            values = sorted(data[col].dropna().unique().tolist(), key=str)
            lines.append(f"@attribute {_quote(col)} {{{','.join(_quote(v) for v in values)}}}")
    return "\n".join(lines + ["", "@data", ""])


def write_arff(path, header, data):
    def cell(v):
        if pd.isna(v):
            return "?"
        return str(v) if isinstance(v, (int, float, np.number)) else _quote(v)
    rows = (",".join(cell(v) for v in row) for row in data.itertuples(index=False))
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")


def _java(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Weka failed:\n{result.stderr or result.stdout}")
    return result.stdout


def run_weka(jar, classifier, train_file, test_file, options, java="java"):
    """Train a Weka classifier on train_file; return (predictions for test_file,
    model printout). Two calls: Weka prints only the
    predictions when -classifications is used (rows like 12,1:e,2:p,+,0.97,
    or 12,1:e,?,... when no class is predicted), so the model is printed by
    a second run."""
    base = [java, "-cp", str(jar), classifier, "-t", str(train_file)]

    evaluation = _java(base + ["-T", str(test_file), "-v", "-classifications",
                               "weka.classifiers.evaluation.output.prediction.CSV", *options])
    predictions = [m.group(1).split(":", 1)[-1] for m in
                   re.finditer(r"^\s*\d+,\d+:[^,]*,(\d+:[^,]*|\?),", evaluation, re.MULTILINE)]
    if not predictions:
        raise RuntimeError(f"No predictions found in {classifier} output:\n" + evaluation[:2000])
    return predictions, _java(base + ["-no-cv", "-v", *options])


def j48_tree(model):
    """Tree listing from a J48 model printout."""
    tree = re.search(r"J48 (?:un)?pruned tree\s*\n-+\n(.*?)\nNumber of Leaves", model, re.DOTALL)
    if tree is None:
        raise RuntimeError("No tree found in J48 output:\n" + model[:2000])
    return tree.group(1).strip()


def id3_tree(model):
    """Tree listing from an Id3 model printout."""
    tree = re.search(r"Id3\s*\n(.*?)\n\s*Time taken", model, re.DOTALL)
    if tree is None:
        raise RuntimeError("No tree found in Id3 output:\n" + model[:2000])
    return tree.group(1).strip()


def tree_text_size(text):
    """Nodes, leaves and depth of a Weka tree listing: one line per branch,
    a line with ':' ends in a leaf; a tree that is a single leaf is ': class'."""
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) == 1 and lines[0].lstrip().startswith(":"):
        return {"nodes": 1, "leaves": 1, "depth": 0}
    leaf_lines = [line for line in lines if ":" in line]
    return {"nodes": len(lines) + 1, "leaves": len(leaf_lines),
            "depth": max(line.count("|") for line in leaf_lines) + 1}


def build_time(model):
    """Time of building the model in seconds, as printed by Weka."""
    found = re.search(r"Time taken to build model\s*:\s*([\d.]+)\s*seconds", model)
    if found is None:
        raise RuntimeError("No build time found in Weka output:\n" + model[:2000])
    return float(found.group(1))


def run_j48(jar, train_file, test_file, options, java="java"):
    """J48 trained on train_file: (predictions for test_file, leaves, tree text)."""
    predictions, model = run_weka(jar, "weka.classifiers.trees.J48",
                                  train_file, test_file, options, java)
    tree = j48_tree(model)
    return predictions, tree_text_size(tree)["leaves"], tree


def run_id3(jar, train_file, test_file, java="java"):
    """Weka Id3 trained on train_file: (predictions for test_file, tree text)."""
    predictions, model = run_weka(jar, "weka.classifiers.trees.Id3",
                                  train_file, test_file, [], java)
    return predictions, id3_tree(model)


def load(dataset, missing_as_nan=True):
    """Dataset with the class as the last column."""
    if dataset == "iris":
        data = bm.load_iris()
    else:
        data = bm.LOADERS[dataset](missing_as_nan=missing_as_nan)
    return data[[c for c in data.columns if c != bm.TARGET] + [bm.TARGET]]


def compare(data, run_treeclass, run_reference, k, seed):
    """Accuracy, agreement and leaves of TreeClass and a Weka classifier on the
    same folds. run_reference(train_file, test_file) returns (predictions, leaves)."""
    features = [c for c in data.columns if c != bm.TARGET]
    header = arff_header(data)
    folds = bm.stratified_folds(data[bm.TARGET], k, seed)

    acc_tc, acc_weka, agree, leaves_tc, leaves_weka = [], [], [], [], []
    with tempfile.TemporaryDirectory() as tmp:
        train_file, test_file = Path(tmp) / "train.arff", Path(tmp) / "test.arff"
        for f in range(k):
            train, test = data[folds != f], data[folds == f]
            write_arff(train_file, header, train)
            write_arff(test_file, header, test)

            result = run_treeclass(train, test, features)
            pred_tc, n_tc = result["predictions"], result["leaves"]
            pred_weka, n_weka = run_reference(train_file, test_file)

            truth = test[bm.TARGET].astype(str).to_numpy()
            pred_tc = np.asarray(pred_tc, dtype=str)
            pred_weka = np.asarray(pred_weka, dtype=str)
            acc_tc.append(np.mean(pred_tc == truth))
            acc_weka.append(np.mean(pred_weka == truth))
            agree.append(np.mean(pred_tc == pred_weka))
            leaves_tc.append(n_tc)
            leaves_weka.append(n_weka)

    return {"accuracy_treeclass": np.mean(acc_tc), "accuracy_weka": np.mean(acc_weka),
            "agreement": np.mean(agree), "leaves_treeclass": np.mean(leaves_tc),
            "leaves_weka": np.mean(leaves_weka) if None not in leaves_weka else np.nan}


def normalize_tree(text):
    """Common form of a Weka and a TreeClass listing: indentation '| ',
    no ':' after internal nodes and no '.0' in whole numbers."""
    lines = []
    for line in text.strip().splitlines():
        line = re.sub(r"\|\s*", "| ", line.rstrip())
        line = re.sub(r"(\d+)\.0+\b", r"\1", line)
        lines.append(line[:-1] if line.endswith(":") else line)
    return "\n".join(lines)


def normalize_id3_tree(text):
    """As normalize_tree, and in the form of Weka Id3, which prints no case
    counts and 'null' for a branch reached by no training case."""
    lines = []
    for line in normalize_tree(text).splitlines():
        line = re.sub(r": \S+ \(0\)$", ": null", line)
        lines.append(re.sub(r" \([^)]*\)$", "", line))
    return "\n".join(lines)


def save_full_trees(dataset, jar, out_dir, java="java"):
    """C4.5 trees of TreeClass and J48 built on the whole dataset, for every
    variant. Saved to out_dir; returns {variant: True if both trees are the same}."""
    data = load(dataset)
    features = [c for c in data.columns if c != bm.TARGET]
    raw, pruned = bm.c45.fit(data, bm.TARGET, features)
    treeclass_trees = {"pruned": pruned, "unpruned": raw}
    same = {}
    with tempfile.TemporaryDirectory() as tmp:
        full = Path(tmp) / "full.arff"
        write_arff(full, arff_header(data), data)
        for variant, (options, _) in VARIANTS.items():
            _, _, j48_tree = run_j48(jar, full, full, options, java)
            treeclass_tree = viz.tree_to_text(treeclass_trees[variant])
            (out_dir / f"{dataset}_{variant}_tree_j48.txt").write_text(j48_tree + "\n", encoding="utf-8")
            (out_dir / f"{dataset}_{variant}_tree_treeclass.txt").write_text(treeclass_tree, encoding="utf-8")
            same[variant] = normalize_tree(j48_tree) == normalize_tree(treeclass_tree)
    return same


def save_full_id3_trees(dataset, jar, out_dir, java="java"):
    """ID3 trees of TreeClass and Weka Id3 built on the whole dataset.
    Saved to out_dir; returns True if both trees are the same."""
    data = load(dataset, missing_as_nan=False)
    features = [c for c in data.columns if c != bm.TARGET]
    treeclass_tree = viz.tree_to_text(bm.id3.fit(data, bm.TARGET, features))
    with tempfile.TemporaryDirectory() as tmp:
        full = Path(tmp) / "full.arff"
        write_arff(full, arff_header(data), data)
        _, weka_tree = run_id3(jar, full, full, java)
    (out_dir / f"{dataset}_id3_tree_weka.txt").write_text(weka_tree + "\n", encoding="utf-8")
    (out_dir / f"{dataset}_id3_tree_treeclass.txt").write_text(treeclass_tree, encoding="utf-8")
    return normalize_id3_tree(weka_tree) == normalize_id3_tree(treeclass_tree)


def main(argv=None):
    ap = argparse.ArgumentParser(description="TreeClass vs. Weka J48 and Id3")
    ap.add_argument("--datasets", nargs="+", default=["iris", "mushroom", "votes"])
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--java", default="java", help="Java executable")
    ap.add_argument("--download-only", action="store_true", help="only fetch the Weka jars")
    ap.add_argument("--out", default=str(bm.ROOT / "output"))
    args = ap.parse_args(argv)

    jar = ensure_weka()
    if args.download_only:
        return

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    rows = []
    for dataset in args.datasets:
        same = save_full_trees(dataset, jar, out_dir, args.java)
        for variant, (options, treeclass_method) in VARIANTS.items():
            def j48(train_file, test_file, options=options):
                predictions, leaves, _ = run_j48(jar, train_file, test_file, options, args.java)
                return predictions, leaves
            row = compare(load(dataset), treeclass_method, j48, args.folds, args.seed)
            rows.append({"dataset": dataset, "variant": f"C4.5 {variant} vs J48",
                         **row, "same_tree": same[variant]})
            print(f"{dataset}, {rows[-1]['variant']}: same_tree = {rows[-1]['same_tree']}")

        if dataset in ID3_DATASETS:
            def id3(train_file, test_file):
                return run_id3(jar, train_file, test_file, args.java)[0], None
            row = compare(load(dataset, missing_as_nan=False), bm.treeclass_id3, id3,
                          args.folds, args.seed)
            rows.append({"dataset": dataset, "variant": "ID3 vs Id3", **row,
                         "same_tree": save_full_id3_trees(dataset, jar, out_dir, args.java)})
            print(f"{dataset}, {rows[-1]['variant']}: same_tree = {rows[-1]['same_tree']}")

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "j48_validation.csv", index=False)
    print()
    print(table.round(4).to_string(index=False))
    print(f"\nsaved: {out_dir / 'j48_validation.csv'} and {out_dir}/*_tree_*.txt")


if __name__ == "__main__":
    sys.exit(main())