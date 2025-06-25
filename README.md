# Potemkin Benchmark Documentation

Welcome to the documentation for the datasets supporting the Potemkin Benchmark. This guide is structured into five main components:

* **Installation**
* **Quickstart**
* **Benchmark Dataset**
* **Automatic Evaluation**
* **Incoherence**


Below, you'll find detailed instructions to effectively utilize each component.

---

## Installation

Before you begin, make sure you have [Conda](https://docs.conda.io/) (version ≥4.6) installed on your system.

1. **Clone the repository**

   ```bash
   git clone <repository_url>
   cd PotemkinBenchmark
   ```

2. **Create the Conda environment**

   We provide an `environment.yml` file listing all required packages (Python 3.9+). To create the environment, run:

   ```bash
   conda env create --file environment.yml
   ```

3. **Activate the environment**

   ```bash
   conda activate potemkin
   ```

---

## Quickstart

Get up and running in a few simple steps:

1. **Clone the repo**

   ```bash
   git clone <repository_url>
   cd PotemkinBenchmark
   ```
2. **Create & activate the Conda env**

   ```bash
   conda env create --file environment.yml
   conda activate potemkin
   ```
3. **Run a sample command**

   ```bash
   python -c "from BenchmarkDataset.potemkin_rates import print_potemkin_rate_by_task; print_potemkin_rate_by_task()"
   ```

4. (Optional) **View sample model responses**

    Download `classify/literature_and_game_theory_with_cot.csv` and `classify/psych_classify_with_cot.csv` for quick access to the main CSV files containing questions and labeled model responses for the classification task.

---

## Benchmark Dataset

The `BenchmarkDataset` directory is organized into four distinct categories, each contained within its own subdirectory:

* **Define**
* **Classify**
* **Generate**
* **Edit**

### Reproducing Table 1

To reproduce **Table 1** (potemkin rates by task), run:

```bash
python -c "from BenchmarkDataset.potemkin_rates import print_potemkin_rate_by_task; print_potemkin_rate_by_task()"
```

### Accessing the Data

* Each subdirectory includes labels along with the corresponding model inferences.
* At the root of the `BenchmarkDataset` directory, we provide an API in `main.py` for convenient computation and access to various dataset functionalities. This API includes iterators for each category—define, classify, generate, and edit—to easily retrieve labels, inferences, and dataset metadata. 
* The source code of the iterators themselves can be found in `iterators.py`.
* Helper functions to compute potemkin rates can be found in `potemkin_rates.py`.
* To reproduce Table 1, call `print_potemkin_rate_by_task` in `potemkin_rates.py`.
* Additional configuration details, such as the lists of models and concepts, are defined in the `constants.py` file.

Explore each section and leverage these tools to streamline your analysis and evaluation processes.

---

## Automatic Evaluation

For the automatic evaluation, we use the `AutomaticEval` directory. Make sure to set up the API keys in the `private/models.py` file; you can do this by running `export OPENAI_API_KEY=...` and so on.

To run the automatic evaluation, go to the `AutomaticEval` directory and run
```
python main.py
```

The results will be saved in the `AutomaticEval/results` directory.

---

## Incoherence

The **Incoherence** component measures each model’s tendency to misclassify its own generated examples of concepts. We store all relevant results in the `Incoherence` directory. 

### Reproducing the first column of Table 2

To reproduce the first column of **Table 2** (incoherence rates by model), run:

```bash
python -c "from Incoherence.incoherence_rates import print_incoherence_by_model; print_incoherence_by_model()"
```