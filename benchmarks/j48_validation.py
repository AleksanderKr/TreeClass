"""
Level 3 validation against the reference implementation Weka J48 (C4.5).

J48 is configured as close as possible to TreeClass:
  -C 0.25   pruning confidence (CF)          -M 1   min. cases per branch
  -S        no subtree raising               -O     do not collapse the tree
  -J        no MDL correction (Release 7)    -doNotMakeSplitPointActualValue
                                                    thresholds at midpoints
Remaining known difference: J48 chooses numeric thresholds by gain
(Release 8), TreeClass by gain ratio (Release 7). For purely categorical
data (Mushroom) the two should therefore build the same trees.

Both tools get the same stratified folds as benchmarks/benchmark.py.
Reported per dataset and variant: accuracy of TreeClass and J48, the share
of test cases on which both predict the same class (agreement), and leaves.
The trees built on the full data are saved to output/ for side-by-side reading.

Requires Java. weka.jar is downloaded from Maven Central on first use
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

import benchmark as bm      # same data loaders, folds and TreeClass methods
import visualize as viz

MAVEN = "https://repo1.maven.org/maven2/nz/ac/waikato/cms/weka"
WEKA_VERSION = "3.8.6"
LIB = Path(__file__).resolve().parent / "lib"
JARS = {   # file: URL (weka-stable needs the "bounce" library at run time)
    "weka.jar": f"{MAVEN}/weka-stable/{WEKA_VERSION}/weka-stable-{WEKA_VERSION}.jar",
    "bounce.jar": f"{MAVEN}/thirdparty/bounce/0.18/bounce-0.18.jar",
}

MATCHED = ["-C", "0.25", "-M", "1", "-S", "-O", "-J", "-doNotMakeSplitPointActualValue"]
VARIANTS = {   # name: (J48 options, TreeClass method)
    "pruned": (MATCHED, bm.treeclass_c45(prune=True)),
    "unpruned": (["-U", "-M", "1", "-O", "-J", "-doNotMakeSplitPointActualValue"],
                 bm.treeclass_c45(prune=False)),
}


# ---------------------------------------------------------------------------
# Weka and ARFF
# ---------------------------------------------------------------------------

def ensure_weka():
    """Download the jars on first use and return the Java classpath."""
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
        raise RuntimeError(f"J48 failed:\n{result.stderr or result.stdout}")
    return result.stdout


def run_j48(jar, train_file, test_file, options, java="java"):
    """Train J48 on train_file; return (predictions for test_file, leaves, tree text).
    Two calls: Weka prints only the predictions when -classifications is used,
    so the model (tree, number of leaves) is printed by a second run."""
    base = [java, "-cp", str(jar), "weka.classifiers.trees.J48", "-t", str(train_file)]

    out = _java(base + ["-T", str(test_file), "-v", "-classifications",
                        "weka.classifiers.evaluation.output.prediction.CSV", *options])
    # prediction rows look like: 12,1:e,2:p,+,0.97
    predictions = [m.group(1) for m in
                   re.finditer(r"^\s*\d+,\d+:[^,]*,\d+:([^,]*),", out, re.MULTILINE)]
    if not predictions:
        raise RuntimeError("No predictions found in J48 output:\n" + out[:2000])

    out = _java(base + ["-no-cv", "-v", *options])
    leaves = re.search(r"Number of Leaves\s*:\s*(\d+)", out)
    if leaves is None:
        raise RuntimeError("No tree found in J48 output:\n" + out[:2000])
    tree = re.search(r"J48 (?:un)?pruned tree\s*\n-+\n(.*?)\nNumber of Leaves", out, re.DOTALL)
    return predictions, int(leaves.group(1)), tree.group(1).strip() if tree else ""

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def load(dataset):
    return bm.load_iris() if dataset == "iris" else bm.load_mushroom(missing_as_nan=True)


def compare(dataset, variant, k, seed, jar, java="java"):
    data = load(dataset)
    data = data[[c for c in data.columns if c != bm.TARGET] + [bm.TARGET]]   # class last
    features = [c for c in data.columns if c != bm.TARGET]
    j48_options, treeclass_method = VARIANTS[variant]
    header = arff_header(data)
    folds = bm.stratified_folds(data[bm.TARGET], k, seed)

    acc_tc, acc_j48, agree, leaves_tc, leaves_j48 = [], [], [], [], []
    with tempfile.TemporaryDirectory() as tmp:
        train_file, test_file = Path(tmp) / "train.arff", Path(tmp) / "test.arff"
        for f in range(k):
            train, test = data[folds != f], data[folds == f]
            write_arff(train_file, header, train)
            write_arff(test_file, header, test)

            pred_tc, n_tc = treeclass_method(train, test, features)
            pred_j48, n_j48, _ = run_j48(jar, train_file, test_file, j48_options, java)

            truth = test[bm.TARGET].astype(str).to_numpy()
            pred_tc = np.asarray(pred_tc, dtype=str)
            pred_j48 = np.asarray(pred_j48, dtype=str)
            acc_tc.append(np.mean(pred_tc == truth))
            acc_j48.append(np.mean(pred_j48 == truth))
            agree.append(np.mean(pred_tc == pred_j48))
            leaves_tc.append(n_tc)
            leaves_j48.append(n_j48)

    return {"dataset": dataset, "variant": variant,
            "accuracy_treeclass": np.mean(acc_tc), "accuracy_j48": np.mean(acc_j48),
            "agreement": np.mean(agree),
            "leaves_treeclass": np.mean(leaves_tc), "leaves_j48": np.mean(leaves_j48)}


def save_full_trees(dataset, jar, out_dir, java="java"):
    """Trees of both tools built on the whole dataset (pruned variant)."""
    data = load(dataset)
    data = data[[c for c in data.columns if c != bm.TARGET] + [bm.TARGET]]
    features = [c for c in data.columns if c != bm.TARGET]
    with tempfile.TemporaryDirectory() as tmp:
        full = Path(tmp) / "full.arff"
        write_arff(full, arff_header(data), data)
        _, _, j48_tree = run_j48(jar, full, full, MATCHED, java)
    _, pruned = bm.c45.fit(data, bm.TARGET, features)
    (out_dir / f"{dataset}_tree_j48.txt").write_text(j48_tree + "\n", encoding="utf-8")
    (out_dir / f"{dataset}_tree_treeclass.txt").write_text(viz.tree_to_text(pruned), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description="TreeClass vs. Weka J48")
    ap.add_argument("--datasets", nargs="+", default=["iris", "mushroom"])
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--java", default="java", help="Java executable")
    ap.add_argument("--download-only", action="store_true", help="only fetch weka.jar")
    ap.add_argument("--out", default=str(bm.ROOT / "output"))
    args = ap.parse_args(argv)

    jar = ensure_weka()
    if args.download_only:
        return

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    rows = []
    for dataset in args.datasets:
        for variant in VARIANTS:
            rows.append(compare(dataset, variant, args.folds, args.seed, jar, args.java))
            print(rows[-1])
        save_full_trees(dataset, jar, out_dir, args.java)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "j48_validation.csv", index=False)
    print()
    print(table.round(4).to_string(index=False))
    print(f"\nsaved: {out_dir / 'j48_validation.csv'} and {out_dir}/*_tree_*.txt")


if __name__ == "__main__":
    sys.exit(main())