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
  - [🤖 Usage](#-usage)
    - [Data process](#data-process)
    - [Train](#train)
    - [Evaluation](#evaluation)
    - [Inference](#inference)
  - [🧪 Tests](#-tests)
- [📌 Project Roadmap](#-project-roadmap)
- [🤝 Contributing](#-contributing)

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
    │   ├── dataproc-cfg.yaml
    |   ├── inference-cfg.yaml
    │   └── train-cfg.yaml
    ├── femo
    │   ├── __init__.py
    │   ├── __version__.py
    │   ├── data
    │   │   ├── __init__.py
    │   │   ├── _utils.py
    │   │   ├── dataset.py
    │   │   ├── pipeline.py
    │   │   ├── ranking.py
    │   │   └── transforms
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
    │   ├── inference.py
    |   ├── inference.sh
    │   ├── process.py
    │   └── test.sh
    │   └── train.py
    └── tests
        ├── requirements.txt
        ├── test_transforms.py
        └── utils.py
```

---

## 🧩 Modules

<details closed><summary>.github.workflows</summary>

| File | Summary |
| --- | --- |
| [ci-check.yml](https://github.com/MAIMLab/FeMo_Analysis/blob/main/.github/workflows/ci-check.yml) | <code>CI pipeline workflow template</code> |

</details>

<details closed><summary>configs</summary>

| File | Summary |
| --- | --- |
| [train-cfg.yaml](https://github.com/MAIMLab/FeMo_Analysis/blob/main/configs/train-cfg.yaml) | <code>Configuration template for training job</code> |
| [inference-cfg.yaml](https://github.com/MAIMLab/FeMo_Analysis/blob/main/configs/inference-cfg.yaml) | <code>Configuration template for inference job</code> |
| [dataproc-cfg.yaml](https://github.com/MAIMLab/FeMo_Analysis/blob/main/configs/dataproc-cfg.yaml) | <code>Configuration template for data processing job</code> |
| [dataManifest.json.template](https://github.com/MAIMLab/FeMo_Analysis/blob/main/configs/dataManifest.json.template) | <code>JSON template for data manifest</code> |
</details>

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

<details closed><summary>scripts</summary>

| File | Summary |
| --- | --- |
| [test.sh](https://github.com/MAIMLab/FeMo_Analysis/blob/main/scripts/test.sh) | <code>Bash script for running linting and pytests</code> |
| [analysis.sh](https://github.com/MAIMLab/FeMo_Analysis/blob/main/scripts/analysis.sh) | <code>Bash script for running data processing and training job</code> |
| [inference.sh](https://github.com/MAIMLab/FeMo_Analysis/blob/main/scripts/inference.sh) | <code>Bash script for running an inference job</code> |
| [inference.py](https://github.com/MAIMLab/FeMo_Analysis/blob/main/femo_analysis/inference.py) | <code>Python script for an inference job</code> |
| [evaluate.py](https://github.com/MAIMLab/FeMo_Analysis/blob/main/femo_analysis/evaluate.py) | <code>Python script for evaluating a trained classifier</code> |
| [train.py](https://github.com/MAIMLab/FeMo_Analysis/blob/main/femo_analysis/train.py) | <code>Python script for a training job</code> |
| [process.py](https://github.com/MAIMLab/FeMo_Analysis/blob/main/femo_analysis/process.py) | <code>Python script for a data processing job</code> |

</details>

<details closed><summary>aws_sagemaker</summary>

| File | Summary |
| --- | --- |
| [inference.py](https://github.com/MAIMLab/FeMo_Analysis/blob/main/aws_sagemaker/inference.py) | <code>TODO implementation</code> |
| [process.py](https://github.com/MAIMLab/FeMo_Analysis/blob/main/aws_sagemaker/process.py) | <code>TODO implementation</code> |
| [transform.py](https://github.com/MAIMLab/FeMo_Analysis/blob/main/aws_sagemaker/transform.py) | <code>TODO implementation</code> |

</details>

<details closed><summary>aws_sagemaker.ml_pipeline</summary>

| File | Summary |
| --- | --- |
| [run_pipeline.sh](https://github.com/MAIMLab/FeMo_Analysis/blob/main/aws_sagemaker/ml_pipeline/run_pipeline.sh) | <code>TODO implementation</code> |
| [run_pipeline.py](https://github.com/MAIMLab/FeMo_Analysis/blob/main/aws_sagemaker/ml_pipeline/run_pipeline.py) | <code>TODO implementation</code> |

</details>

---

## 🚀 Getting Started

### 🔖 Prerequisites

**Python**: `version 3.10`

### 📦 Installation

Create a new `conda` or `virtualenv` with appropriate Python version

Build the project from source:

1. Clone the FeMo_Analysis repository:
```sh
❯ git clone https://github.com/MAIMLab/FeMo_Analysis
```
For cloning using SSH, make sure to create and store SSH key on your device. Then, clone the repository:
```sh
❯ git clone git@github.com:MAIMLab/FeMo_Analysis.git
```

2. Navigate to the project directory:
```sh
❯ cd FeMo_Analysis
```

3. Install the repo as a package:
```sh
❯ pip install -e .
```

### 🤖 Usage

*When running python scripts, first activate appropriate virtual environment. Bash scripts for a particular job automatically creates
environment with necessary dependencies.*

#### Data process
```sh
❯ python scripts/process.py [-h] [--data-dir DATA_DIR] [--params-filename PARAMS_FILENAME] [--work-dir WORK_DIR] [--extract] dataManifest

positional arguments:
  dataManifest          Path to data manifest json file

options:
  -h, --help            show this help message and exit
  --data-dir DATA_DIR   Path to directory containing .dat and .csv files ( default ./data )
  --params-filename PARAMS_FILENAME
                        Parameters dict filename ( default params_dict.pkl)
  --work-dir WORK_DIR   Path to save generated artifacts ( default ./work_dir/ )
  --extract             Extract features
```

#### Train
```sh
❯ python scripts/train.py [-h] [--tune] [--work-dir WORK_DIR] datasetDir ckptName

positional arguments:
  datasetDir           Directory containing train test pickle files
  ckptName             Name of model checkpoint file

options:
  -h, --help           show this help message and exit
  --tune               Tune hyperparameters before training
  --work-dir WORK_DIR  Path to save generated artifacts
```

#### Evaluation
```sh
❯ python scripts/evaluate.py [-h] [--data-dir DATA_DIR] [--work-dir WORK_DIR] [--outfile OUTFILE] dataManifest resultsDir

positional arguments:
  dataManifest         Path to data manifest json file
  resultsDir           Directory containing prediction results

options:
  -h, --help           show this help message and exit
  --data-dir DATA_DIR  Path to directory containing .dat and .csv files
  --work-dir WORK_DIR  Path to save generated artifacts
  --outfile OUTFILE    Metrics output file
```

Together, an **analysis job** (process -> train -> evaluate) can be run with following command:
```sh
❯ bash scripts/analysis.sh <data_manifest> <ckpt_name> [output_file] [run_dir] [params_filename]
```

#### Inference
```sh
❯ python scripts/inference.py [-h] [--work-dir WORK_DIR] [--outfile OUTFILE] dataFilename ckptFilename paramsFilename

positional arguments:
  dataFilename         Path to data file
  ckptFilename         Name of model checkpoint file
  paramsFilename       Parameters dict filename

options:
  -h, --help           show this help message and exit
  --work-dir WORK_DIR  Path to save generated artifacts
  --outfile OUTFILE    Metrics output file
```
To run an inference job using `bash`, run the following command:
```sh
❯ bash scripts/inference.sh <data_filename> <ckpt_name> <params_dict> [output_file] [run_dir]
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
