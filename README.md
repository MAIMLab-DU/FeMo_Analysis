<p align="center">
    <h1 align="center">FEMO_ANALYSIS</h1>
</p>
<p align="center">
    <em><code>Fetal Movement Monitor and Analysis</code></em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/MAIMLab/FeMo_Analysis?style=plastic&logo=opensourceinitiative&logoColor=white&color=0c8106" alt="license">
	<img src="https://img.shields.io/github/last-commit/MAIMLab/FeMo_Analysis?style=plastic&logo=git&logoColor=white&color=0c8106" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/MAIMLab/FeMo_Analysis?style=plastic&color=0c8106" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/MAIMLab/FeMo_Analysis?style=plastic&color=0c8106" alt="repo-language-count">
</p>
<p align="center">
		<em>Built with the tools and technologies:</em>
</p>
<p align="center">
	<img src="https://img.shields.io/badge/GNU%20Bash-4EAA25.svg?style=plastic&logo=GNU-Bash&logoColor=white" alt="GNU%20Bash">
	<img src="https://img.shields.io/badge/TensorFlow-FF6F00.svg?style=plastic&logo=TensorFlow&logoColor=white" alt="TensorFlow">
	<img src="https://img.shields.io/badge/scikitlearn-F7931E.svg?style=plastic&logo=scikit-learn&logoColor=white" alt="scikitlearn">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=plastic&logo=Python&logoColor=white" alt="Python">
	<img src="https://img.shields.io/badge/GitHub%20Actions-2088FF.svg?style=plastic&logo=GitHub-Actions&logoColor=white" alt="GitHub%20Actions">
</p>

<br>

<details><summary>Table of Contents</summary>

- [📍 Overview](#-overview)
- [👾 Features](#-features)
- [📂 Repository Structure](#-repository-structure)
- [🧩 Modules](#-modules)
- [🚀 Getting Started](#-getting-started)
  - [🔖 Prerequisites](#-prerequisites)
  - [📦 Installation](#-installation)
      - [Optional](#optional)
  - [🤖 Usage](#-usage)
    - [Feature Extraction](#feature-extraction)
    - [Data Preprocessing](#data-preprocessing)
    - [Train](#train)
    - [Evaluation](#evaluation)
    - [Inference](#inference)
  - [🧪 Tests](#-tests)
- [📌 Project Roadmap](#-project-roadmap)
- [🤝 Contributing](#-contributing)
  - [⚠️ **Caution**](#️-caution)

</details>
<hr>

## 📍 Overview

This repository contains the Python codebase for the FeMo (Fetal Movement) device Version 2, developed to monitor and analyze fetal movements. Fetal movement patterns provide valuable insights for both medical research and clinical applications, helping in assessing fetal health during pregnancy. This project focuses on processing raw data captured by the FeMo device and applying various analytical techniques to extract meaningful information regarding fetal activity.

The FeMo device collects movement data continuously from the fetus, and this repository contains the necessary scripts and tools to handle, clean, and analyze this data. The aim is to facilitate detailed fetal movement studies that can contribute to medical diagnostics, early detection of potential issues, and overall fetal well-being monitoring.

---

## 👾 Features

* **Data Processing:** Scripts for handling and processing raw data from the FeMo device.
* **Analysis:** Analytical tools and algorithms to evaluate fetal movement patterns.
* **AWS Integration:** Configured for use with AWS SageMaker, providing scalability for data processing.
* **Testing Suite:** Includes unit tests to ensure code reliability.

---

## 📂 Repository Structure

```sh
└── FeMo_Analysis/
    ├── .github
    │   ├── ISSUE_TEMPLATE
    │   │   ├── bug_report.md
    │   │   └── feature_request.md
    │   └── workflows
    │       └── ci-check.yml
    ├── README.md
    ├── aws_sagemaker
    │   ├── inference.py
    │   ├── ml_pipeline
    │   │   ├── run_pipeline.py
    │   │   └── run_pipeline.sh
    │   ├── process.py
    │   └── transform.py
    ├── configs
    │   ├── dataManifest.json.template
    │   ├── dataset-cfg.yaml
    │   ├── preprocess-cfg.yaml
    │   ├── inference-cfg.yaml
    │   └── train-cfg.yaml
    ├── femo
    │   ├── __init__.py
    │   ├── __version__.py
    │   ├── data
    │   │   ├── __init__.py
    │   │   ├── _utils.py
    │   │   ├── dataset.py
    │   │   ├── pipeline.py
    │   │   ├── preprocess.py
    │   │   ├── ranking.py
    │   │   └── transforms
    │   │       └── ... 
    │   ├── eval
    │   │   ├── __init__.py
    │   │   └── metrics.py
    │   ├── logger
    │   │   ├── __init__.py
    │   │   └── _utils.py
    │   └── model
    │       ├── __init__.py
    │       ├── adaboost.py
    │       ├── base.py
    │       ├── ensemble.py
    │       ├── femonet.py
    │       ├── log_regression.py
    │       ├── random_forest.py
    │       └── svm.py
    ├── pytest.ini
    ├── requirements.txt
    ├── scripts
    │   ├── analysis.sh
    │   ├── evaluate.py
    │   ├── extract.py
    │   ├── inference.py
    │   ├── inference.sh
    │   ├── preprocess.py
    │   ├── test.sh
    │   └── train.py
    └── tests
        ├── requirements.txt
        ├── test_transforms.py
        └── utils.py
```

---

## 🧩 Modules

<details closed><summary>femo</summary>

| File | Summary |
| --- | --- |
| [__version__.py](https://github.com/MAIMLab/FeMo_Analysis/blob/main/femo/__version__.py) | <code>Version of the project</code> |

</details>

<details closed><summary>femo.model</summary>

| File | Summary |
| --- | --- |
| [base.py](https://github.com/MAIMLab/FeMo_Analysis/blob/main/femo/model/base.py) | <code>Module containing FeMoBaseClassifier</code> |
| [adaboost.py](https://github.com/MAIMLab/FeMo_Analysis/blob/main/femo/model/adaboost.py) | <code>Module containing FeMoAdaBoostClassifier</code> |
| [log_regression.py](https://github.com/MAIMLab/FeMo_Analysis/blob/main/femo/model/log_regression.py) | <code>Module containing FeMoLogRegClassifier</code> |
| [femonet.py](https://github.com/MAIMLab/FeMo_Analysis/blob/main/femo/model/femonet.py) | <code>Module containing FeMoNNClassifier</code> |
| [random_forest.py](https://github.com/MAIMLab/FeMo_Analysis/blob/main/femo/model/random_forest.py) | <code>Module containing FeMoRFClassifier</code> |
| [svm.py](https://github.com/MAIMLab/FeMo_Analysis/blob/main/femo/model/svm.py) | <code>Module containing FeMoSVClassifier</code> |
| [ensemble.py](https://github.com/MAIMLab/FeMo_Analysis/blob/main/femo/model/ensemble.py) | <code>Module containing FeMoEnsembleClassifier</code> |

</details>

<details closed><summary>femo.logger</summary>

| File | Summary |
| --- | --- |
| [_utils.py](https://github.com/MAIMLab/FeMo_Analysis/blob/main/femo/logger/_utils.py) | <code>Utility functions for logger</code> |

</details>

<details closed><summary>femo.eval</summary>

| File | Summary |
| --- | --- |
| [metrics.py](https://github.com/MAIMLab/FeMo_Analysis/blob/main/femo/eval/metrics.py) | <code>Module containing FeMoMetrics class for calculating necessary metrics</code> |

</details>

---

## 🚀 Getting Started
This codebase is developed and tested on **Ubuntu 22.04** operating system and **Python 3.10.15**. For running *Shell Script* file in **Windows** operating system, please follow [this guide](https://www.thewindowsclub.com/how-to-run-sh-or-shell-script-file-in-windows-10).

### 🔖 Prerequisites

**Python**: `version 3.10.15`.
**conda** or **virtualenv** must be installed on system.

### 📦 Installation

Create a new `conda` or `virtualenv`.

Build the project from source:

1. Clone the FeMo_Analysis repository:
```sh
❯ git clone https://github.com/MAIMLab/FeMo_Analysis
```
For cloning using SSH, make sure to create and store SSH key on your device. Then, clone the repository:
```sh
❯ git clone git@github.com:MAIMLab/FeMo_Analysis.git
```

1. Navigate to the project directory:
```sh
❯ cd FeMo_Analysis
```

1. Install the repo as a package:
```sh
❯ pip install .
```
##### Optional
a. To install the package for development:
```sh
❯ pip install .[dev]
```
b. To install the package for integration with AWS Sagemaker:
```sh
❯ pip install .[sagemaker]
```

### 🤖 Usage

*When running python scripts, first activate appropriate virtual environment. Bash scripts for a particular job automatically creates
environment with necessary dependencies.*

**WARNING: Running bash scripts in Windows OS will require installing [Git bash](https://git-scm.com/downloads/win)**

#### Feature Extraction
```sh
❯ python scripts/extract.py [-h] --data-manifest DATA_MANIFEST [--data-dir DATA_DIR] [--work-dir WORK_DIR] [--config-path CONFIG_PATH] [--extract]

options:
  -h, --help            show this help message and exit
  --data-manifest DATA_MANIFEST
                        Path to data manifest json file
  --data-dir DATA_DIR   Path to directory containing .dat and .csv files
  --work-dir WORK_DIR   Path to save generated artifacts
  --config-path CONFIG_PATH
                        Path to config file
  --extract             Force extract features even if they exist
```

#### Data Preprocessing
```sh
❯ python scripts/process.py [-h] --features-dir FEATURES_DIR [--work-dir WORK_DIR] [--config-path CONFIG_PATH] [--train-config-path TRAIN_CONFIG_PATH]

options:
  -h, --help            show this help message and exit
  --features-dir FEATURES_DIR
                        Directory containing 'features.csv' file
  --work-dir WORK_DIR   Path to save generated artifacts
  --config-path CONFIG_PATH
                        Path to config file
  --train-config-path TRAIN_CONFIG_PATH
                        Path to config file
```

#### Train
```sh
❯ python scripts/train.py [-h] --train TRAIN --model-dir MODEL_DIR --output-data-dir OUTPUT_DATA_DIR [--tune] [--config-path CONFIG_PATH]

options:
  -h, --help            show this help message and exit
  --train TRAIN         Path to dataset
  --model-dir MODEL_DIR
                        Model output directory
  --output-data-dir OUTPUT_DATA_DIR
                        Model output directory
  --tune                Tune hyperparameters before training
  --config-path CONFIG_PATH
                        Path to config file
```

#### Evaluation
```sh
❯ python scripts/evaluate.py [-h] --data-manifest DATA_MANIFEST --results-path RESULTS_PATH --metadata-path METADATA_PATH [--config-path CONFIG_PATH] [--data-dir DATA_DIR] [--work-dir WORK_DIR] [--out-filename OUT_FILENAME]
                   [--sagemaker SAGEMAKER]

options:
  -h, --help            show this help message and exit
  --data-manifest DATA_MANIFEST
                        Path to data manifest json file
  --results-path RESULTS_PATH
                        Path to file containing prediction results
  --metadata-path METADATA_PATH
                        Path to file containing prediction metadata
  --config-path CONFIG_PATH
                        Path to config file
  --data-dir DATA_DIR   Path to directory containing .dat and .csv files
  --work-dir WORK_DIR   Path to save generated artifacts
  --out-filename OUT_FILENAME
                        Metrics output filename
  --sagemaker SAGEMAKER
                        Sagemaker directory for results
```

Together, an **analysis job** (feature_extraction -> process -> train -> repack -> evaluate) can be run with following command:
```sh
❯ bash scripts/analysis.sh [-f|--force-extract] -m <manifest_file> [-d|--dataset-cfg] [-c|preprocess-cfg] [-t|--train-cfg] [-w|--work-dir] [-r|--run-name] [-p|--perf-filename] [-h|--help]

Options:
  -m <manifest_file>, --manifest-file
                             Path to the dataManifest file.
  -f, --force-extract        Force extract features.
  -d <dataset_config>, --dataset-cfg
                             Path to dataset config (default 'dataset-cfg.yaml').
  -c <preproc_config>, --preprocess-cfg
                             Path to preprocess config (default 'preprocess-cfg.yaml').
  -t <train_config>, --train-cfg
                             Path to training config (default 'train-cfg.yaml').
  -w <work_dir>, --work-dir
                             Project working directory.
  -r <run_name>, --run-name
                             Name of a specific run.
  -p <perf_filename>, --perf-filename
                             Performance csv filename.
  -h, --help             Show this help message and exit.
```

#### Inference
```sh
❯ python inference.py [-h] --data-file DATA_FILE --classifier CLASSIFIER --model MODEL --pipeline PIPELINE --processor PROCESSOR --metrics METRICS [--config-path CONFIG_PATH] [--work-dir WORK_DIR] [--outfile OUTFILE] [--remove-hiccups REMOVE_HICCUPS] [--plot PLOT]

options:
  -h, --help            show this help message and exit
  --data-file DATA_FILE
                        Path to data file(s) (.dat or .txt)
  --classifier CLASSIFIER
                        Path to fitted classifier object file (.joblib)
  --model MODEL         Path to trained model file (.joblib or .h5)
  --pipeline PIPELINE   Path to data pipeline object (.joblib)
  --processor PROCESSOR
                        Path to data processor object (.joblib)
  --metrics METRICS     Path to evaluation metrics object (.joblib)
  --config-path CONFIG_PATH
                        Path to config file
  --work-dir WORK_DIR   Path to save generated artifacts
  --outfile OUTFILE     Metrics output file
  --remove-hiccups REMOVE_HICCUPS
                        Exclude hiccups from ML detections map
  --plot PLOT           Generate and save detection plots
```
To run an inference job using `bash`, run the following command:
```sh
❯ bash scripts/inference.sh [-z|--remove-hiccups] [-p|--plot] -d <data_filename> -m <repacked_model> [-w|--work-dir] [-r|--run-name] [-f|--perf-filename] [-h|--help]

Options:
  -d <data_filename>, --data-filename
                             Path to log data file(s) (.dat or .txt).
  -z, --remove-hiccups       Remove hiccups for analysis.
  -p, --plot                 Generate plots.
  -m <repacked_model>, --repacked-model
                             Path to repacked model file (.tar.gz).
  -f <perf_filename>, --perf-filename
                             Performance csv filename.
  -w <work_dir>, --work-dir
                             Project working directory.
  -r <run_name>, --run-name
                             Name of a specific run.
  -h, --help             Show this help message and exit.
```

### 🧪 Tests

Execute the test suite using the following command:

```sh
❯ bash scripts/test.sh
```

---

## 📌 Project Roadmap

- [X] **`Task 1`**: <strike>Implement inference job</strike>
- [ ] **`Task 2`**: Implement AWS Sagemaker integration.

---

## 🤝 Contributing

### ⚠️ **Caution**
Currently, branch protection rules are not enforced; however, contributors are strongly advised **not** to merge pull requests without at least one `Approved` review. Please ensure compliance with the _Contributing Guidelines_ below.

Contributions are welcome! Here are several ways you can contribute:

- **[Report Issues](https://github.com/MAIMLab/FeMo_Analysis/issues)**: Submit bugs found or log feature requests for the `FeMo_Analysis` project.
- **[Submit Pull Requests](https://github.com/MAIMLab/FeMo_Analysis/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/MAIMLab/FeMo_Analysis
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8.  **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/MAIMLab/FeMo_Analysis/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=MAIMLab/FeMo_Analysis">
   </a>
</p>
</details>

---
