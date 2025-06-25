# Potemkin Benchmark Documentation

Welcome to the documentation for the datasets supporting the Potemkin Benchmark. This guide is structured into the following main components:

* **Installation**
* **Quickstart**
* **Benchmark Dataset**
* **Automatic Evaluation**
* **Incoherence**


Below, you'll find detailed instructions to effectively utilize each component.

## Installation

Before you begin, make sure you have [Conda](https://docs.conda.io/) (version ≥4.6) installed on your system.

1. **Clone the repository**

   ```bash
   git clone https://github.com/MarinaMancoridis/PotemkinBenchmark.git
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

## Quickstart

Get up and running in a few simple steps:

1. **Run a sample command**

   ```bash
   python -c "from BenchmarkDataset.potemkin_rates import print_potemkin_rate_by_task; print_potemkin_rate_by_task()"
   ```

2. **View sample model responses**

    Download `classify/literature_and_game_theory_with_cot.csv` or `classify/psych_classify_with_cot.csv` for quick access to the main CSV files containing questions and labeled model responses for the classification task.

---

## Benchmark Dataset

The `BenchmarkDataset` directory is organized by the four main tasks in our framework, each contained within its own subdirectory:

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

* We provide an iterator for each task. The iterator provides access to labeled model responses in a standardized format.
* A sample API for using the iterators is provided in `BenchmarkDataset/main.py`. 
* The source code for the iterators themselves can be accessed in `BenchmarkDataset/iterators.py`.
* Helper functions to compute potemkin rates can be found in `potemkin_rates.py`.
* Additional configuration details, such as the lists of models and concepts, are defined in the `constants.py` file.

## Automatic Evaluation

For the automatic evaluation, we use the `AutomaticEval` directory. Make sure to set up the API keys in the `private/models.py` file; you can do this by running `export OPENAI_API_KEY=...` and so on.

To run the automatic evaluation, go to the `AutomaticEval` directory and run
```
python main.py
```

The results will be saved in the `AutomaticEval/results` directory.

## Incoherence

All relevant results for our incoherence analysis are provided in the `Incoherence` directory. 

### Reproducing the first column of Table 2

To reproduce the first column of **Table 2** (incoherence rates by model), run:

```bash
python -c "from Incoherence.incoherence_rates import print_incoherence_by_model; print_incoherence_by_model()"
```