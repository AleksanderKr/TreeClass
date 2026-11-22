# TreeClass: Decision Tree Induction and Visualization with ID3 and C4.5

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AleksanderKr/TreeClass/HEAD?urlpath=%2Fdoc%2Ftree%2FTreeClass_demo.ipynb)

TreeClass is a Python toolkit for decision tree induction with the ID3 and C4.5 algorithms. It exposes every step of tree induction (entropy, information gain, gain ratio, numeric thresholds, missing-value weights, error-based pruning), prints and draws the induced trees, and is validated against the Weka reference implementations (J48 and Id3).

## Table of Contents
* [Interactive Demonstration (Zero Setup)](#interactive-demonstration-zero-setup)
* [Installation](#installation)
* [Repository Structure](#repository-structure)
* [Input Data Format Specifications](#input-data-format-specifications)
* [Command Line Usage](#command-line-usage)
* [Minimal Working Example (Python API)](#minimal-working-example-python-api)
* [Implemented C4.5 Variant](#implemented-c45-variant)
* [Testing and Reproducibility](#testing-and-reproducibility)
* [Validation against Weka](#validation-against-weka)
* [Benchmarking Suite](#benchmarking-suite)
* [Datasets](#datasets)
* [Outputs](#outputs)
* [License](#license)

## Interactive Demonstration (Zero Setup)

You can run and evaluate the entire toolkit directly in your browser using the preconfigured Jupyter Notebook environment. The notebook reproduces the examples of the software article step by step, the functional verification against Weka J48 and Id3, and the performance benchmarking suite.

[![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AleksanderKr/TreeClass/HEAD?urlpath=%2Fdoc%2Ftree%2FTreeClass_demo.ipynb)

*(Click the badge above to launch the reproducible capsule. The initial environment build may take a few minutes.)*

> Performance Note: Binder environments operate under strict resource constraints (limited RAM and CPU). The full benchmarking suite takes several minutes there; for faster execution, clone the repository and run the notebook locally.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AleksanderKr/TreeClass.git
   cd TreeClass
   ```

2. Ensure you have Python 3.10 or higher installed (tested with Python 3.12):
   ```bash
   python --version
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

*(Note: chefboost is only required for the benchmarking suite. Java is only required for the validation against Weka; the Weka jars are downloaded from Maven Central on first use.)*

## Repository Structure

```
src/            ID3 (id3.py), C4.5 (c45.py), visualization (visualize.py), command line pipeline (pipeline.py)
tests/          tests of the article examples, edge cases and the benchmark datasets
benchmarks/     benchmark against chefboost and Weka (benchmark.py), validation against Weka (j48_validation.py)
data/           Play Tennis variants and the UCI datasets
.binder/        Binder environment configuration
TreeClass_demo.ipynb   interactive demonstration notebook
```

## Input Data Format Specifications

The pipeline reads a CSV file with a header row. One column is the class (target); every other column is an input attribute.
* **Attribute types** are detected automatically: numeric columns get binary threshold tests (`A <= t`) in C4.5, all other columns are categorical.
* **Missing values** are written as `?` or left empty. C4.5 handles them with fractional weights (Quinlan, 1993); ID3 rejects them.
* **Rows without a class label** are removed with a note.
* **Identifier columns** (e.g. a row number) must be excluded with `--exclude`, otherwise ID3 would split on them.

*Example structure (`data/play_tennis_numeric.csv`):*
```csv
Day,Outlook,Temperature,Humidity,Wind,Play
1,Sunny,85,85,Weak,No
2,Sunny,80,90,Strong,No
3,Overcast,83,78,Weak,Yes
```

*(ID3 treats every attribute as categorical, so a numeric column is split into one branch per distinct value.)*

## Command Line Usage

Use the pipeline script `src/pipeline.py` from the repository root. It builds the trees, prints them with their size and accuracy, and saves the drawings to `output/`.

* Both algorithms on the numeric Play Tennis dataset (default):
  ```bash
  python src/pipeline.py
  ```

* ID3 on the categorical Play Tennis dataset:
  ```bash
  python src/pipeline.py --data data/play_tennis.csv --algo id3
  ```

* C4.5 on own data with a held-out test set:
  ```bash
  python src/pipeline.py --data data/my.csv --target class --exclude id --algo c45 --test-size 0.3
  ```

`python src/id3.py` and `python src/c45.py` accept the same options and run one algorithm only.

### Pipeline Command Options (`pipeline.py`)

* **`--data`**: Path to the input CSV file (default: `data/play_tennis_numeric.csv`).
* **`--target`**: Name of the class column (default: `Play`).
* **`--algo`**: Algorithm choice: `id3`, `c45` or `both` (default: `both`).
* **`--exclude`**: Columns to ignore if present, e.g. row identifiers (default: `Day`).
* **`--test-size`**: Fraction of rows held out for testing (default: `0`, train on all rows).
* **`--seed`**: Random seed of the train/test split (default: `42`).
* **`--cf`**: C4.5 pruning confidence factor; smaller values prune more (default: `0.25`).
* **`--min-cases`**: C4.5 minimum (weighted) number of cases in at least two branches of a test (default: `1`).
* **`--out`**: Output directory for the drawings (default: `output`).
* **`--show`**: Also open the drawings in a window.

## Minimal Working Example

Since this package is not distributed via PyPI, execute your scripts from the root directory of the cloned repository and add `src` to the import path.

```python
import sys
import pandas as pd

sys.path.insert(0, "src")
import c45
import id3
import visualize as viz

# 1. Load the data (the Day column is an identifier, not an attribute)
data = pd.read_csv("data/play_tennis_numeric.csv").drop(columns="Day")

# 2. Build a C4.5 tree: fit returns the tree before and after pruning
unpruned, pruned = c45.fit(data, target="Play", cf=0.25)

# 3. Inspect the tree: text listing, size and a drawing
viz.print_tree(pruned)
print(viz.tree_stats(pruned))
viz.plot_tree(pruned, title="C4.5 - Play Tennis", path="output/play_tennis_c45.png")

# 4. Classify cases
print(c45.predict(pruned, data))

# 5. ID3 on the categorical variant
categorical = pd.read_csv("data/play_tennis.csv").drop(columns="Day")
viz.print_tree(id3.fit(categorical, target="Play"))
```

Trees are plain nested Python dictionaries (see the docstrings of `id3.py` and `c45.py`), so they can be inspected, traversed and modified directly.

## Implemented C4.5 Variant

TreeClass follows C4.5 as described in Quinlan (1993), i.e. C4.5 Release 7:
* Tests are chosen by gain ratio among the tests whose gain is at least the average gain.
* Numeric thresholds are chosen by gain ratio at midpoints between adjacent values; there is no MDL penalty (both are changes of Release 8, Quinlan 1996).
* Missing attribute values are handled with fractional case weights during induction and classification.
* Error-based pruning (CF = 25%) is checked at every node, bottom-up; as in Quinlan's code, a subtree is replaced by a leaf when the leaf's predicted errors are at most 0.1 above the subtree's.

Deliberate simplifications:
* The minimum number of cases per branch defaults to 1 (Quinlan's default is 2); it can be changed with `--min-cases` / `min_cases`.
* Subtree raising (replacing a node by its largest branch) is not implemented.
* The upper confidence limit uses the exact normal quantile (as in Weka J48) instead of Quinlan's interpolated table.

## Testing and Reproducibility

Run the full test suite from the repository root:
```bash
python -m pytest
```

To trace every intermediate step of the ID3 and C4.5 examples from the software article (entropy, gain, split information, gain ratio, candidate thresholds, average-gain safeguard, resulting trees), run the article test script directly:
```bash
python tests/test_article_examples.py
```

## Validation against Weka

The script `benchmarks/j48_validation.py` compares TreeClass with the Weka reference implementations on identical stratified folds: C4.5 with J48 (configured as close as possible: `-C 0.25 -M 1 -S -O -J -doNotMakeSplitPointActualValue`) and ID3 with Id3. Besides accuracy and agreement of predictions, the trees built on the full data are compared line by line.

```bash
python benchmarks/j48_validation.py --folds 10
```

On the categorical datasets (Mushroom, Votes) TreeClass builds exactly the same trees as J48 and Id3, before and after pruning. On Iris the trees may differ, because J48 implements Release 8, which chooses numeric thresholds by gain instead of gain ratio.

### Validation Command Options (`j48_validation.py`)

* **`--datasets`**: Datasets to compare: `iris`, `mushroom`, `votes` (default: all).
* **`--folds`**: Number of cross-validation folds (default: `10`).
* **`--seed`**: Random seed of the folds (default: `0`).
* **`--java`**: Java executable (default: `java`).
* **`--download-only`**: Only fetch the Weka jars.
* **`--out`**: Output directory (default: `output`).

## Benchmarking Suite

The script `benchmarks/benchmark.py` evaluates TreeClass, chefboost and Weka with stratified cross-validation and reports accuracy, macro-F1, tree size (nodes, leaves, depth), build time and prediction time.

* Example execution:
  ```bash
  python benchmarks/benchmark.py --folds 10
  ```

* Scalability on growing stratified samples:
  ```bash
  python benchmarks/benchmark.py --datasets mushroom --fractions 0.1 0.25 0.5 1 --out output/benchmark_scaling.csv
  ```

### Benchmark Command Options (`benchmark.py`)

* **`--datasets`**: Datasets to run: `iris`, `mushroom`, `votes` (default: all).
* **`--folds`**: Number of cross-validation folds (default: `10`).
* **`--seed`**: Random seed of the folds and samples (default: `0`).
* **`--no-chefboost`**: Skip chefboost.
* **`--no-weka`**: Skip Weka J48 and Id3 (no Java needed).
* **`--fractions`**: Sizes of the stratified samples, e.g. `0.1 0.25 0.5 1` (default: `1`).
* **`--out`**: Output CSV file (default: `output/benchmark.csv`).

## Datasets

* **Play Tennis** (`data/play_tennis*.csv`): the classic example of Quinlan (1986, 1993), in a categorical and a numeric variant.
* **Iris** (`data/iris.data`), **Mushroom** (`data/mushroom.data`) and **Congressional Voting Records** (`data/house-votes-84.data`): from the [UCI Machine Learning Repository](https://archive.ics.uci.edu), licensed under CC BY 4.0.

## Outputs

* `src/pipeline.py`: tree drawings `output/<data>_id3.png`, `output/<data>_c45_unpruned.png` and `output/<data>_c45_pruned.png`.
* `benchmarks/benchmark.py`: metrics table `output/benchmark.csv`.
* `benchmarks/j48_validation.py`: comparison table `output/j48_validation.csv` and the tree listings of both tools, e.g. `output/votes_pruned_tree_treeclass.txt` and `output/votes_pruned_tree_j48.txt`.

## License

TreeClass is released under the license in [LICENSE](LICENSE). Weka is licensed under the GPL and is not distributed with this repository; it is downloaded from Maven Central when the validation is run.