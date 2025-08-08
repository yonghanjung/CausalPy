# A Benchmark for Estimating Identifiable Causal Effects

**A curated collection of data generating processes from the causal inference literature, designed for benchmarking and evaluating causal effect estimation methods.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 Overview

This repository provides a standardized, easy-to-use collection of synthetic and semi-synthetic data generating processes (DGPs) for the task of estimating any identifiable causal effects from observational data and a causal graph. The goal is to offer a benchmark for evaluating the performance of causal estimators under various identification scenarios, such as

* Back-door adjustment
* Sequential time-series data 
* Front-door adjustment 
* Tian's adjustment 
* Napkin graph 
* Other complicated graphs

Each DGP is implemented as a **StructuralCausalModel (SCM)**, allowing for flexible data generation and intervention.

---

## 💻 Data Generation Script

For maximum reproducibility and flexibility, we provide the Python script (`example_SCM.py`) used to generate all datasets from their underlying SCMs. This allows researchers to generate new samples with different random seeds, sizes, or parameter settings.

### Setup

1.  Make sure you have the necessary libraries installed:
    ```bash
    pip install numpy pandas scipy
    ```
2.  The generation script depends on a custom `StructuralCausalModel` class. Ensure the `SCM.py` file is in the same directory as `data_generators.py`.

### How to Generate Data

You can easily generate data from any SCM in the collection. Each generator function returns the SCM object and lists of the treatment and outcome variable names.

Here is an example of how to generate 1,000 samples from the **Kang & Schafer (2007)** simulation:

```python
# 1. Import the generator function and the SCM class
from data_generators import Kang_Schafer
from SCM import StructuralCausalModel

# 2. Get the SCM object and variable names
scm, treatments, outcomes = Kang_Schafer(seednum=42)

# 3. Generate a pandas DataFrame with 1,000 samples
sample_data = scm.generate_samples(num_samples=1000)

# Display the first few rows
print(sample_data.head())