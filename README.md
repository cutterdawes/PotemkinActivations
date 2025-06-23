# Potemkin Benchmark Documentation

Welcome to the documentation for the datasets supporting the Potemkin Benchmark. This guide is structured into two main components:

* **Benchmark Dataset**
* **Automatic Evaluation**

Below, you'll find detailed instructions to effectively utilize each component.

## Benchmark Dataset

The `BenchmarkDataset` directory is organized into four distinct categories, each contained within its own subdirectory:

* **Define**
* **Classify**
* **Generate**
* **Edit**

### Accessing the Data

* Each subdirectory includes labels along with the corresponding model inferences.
* At the root of the `BenchmarkDataset` directory, we provide an API in `main.py` for convenient computation and access to various dataset functionalities. This API includes iterators for each category—define, classify, generate, and edit—to easily retrieve labels, inferences, and dataset metadata. 
* The source code of the iterators themselves can be found in `iterators.py`.
* Helper functions to compute potemkin rates can be found in `potemkin_rates.py`.
* Additional configuration details, such as the lists of models and concepts, are defined in the `constants.py` file.

Explore each section and leverage these tools to streamline your analysis and evaluation processes.

## Automatic Evaluation

For the automatic evaluation, we use the `automatic-eval` directory. Make sure to set up the API keys in the `private/models.py` file; you can do this by running `export OPENAI_API_KEY=...` and so on.

To run the automatic evaluation, go to the `automatic-eval` directory and run
```
python main.py
```

The results will be saved in the `automatic-eval/results` directory.