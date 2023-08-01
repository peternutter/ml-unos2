# Project README

## Overview

This project comprises two machine learning workflows for survival analysis, using the PySurvival and scikit-survival libraries. These workflows handle the complete data life-cycle from preprocessing, training models, validating them, saving the trained models, and logging events. We have also implemented permutation feature importance for the PySurvival library using the C-Index. You can run these pipelines on a computing cluster.

**Note:** These pipelines are tailored for a specific cluster setup and might require modifications to run on a different configuration.

## Project Structure

The project has the following structure:

```
.
├── config
│   ├── column_names.py
│   ├── config.py
├── pysurv
│   ├── a_submit.sh
│   ├── b_submit.sh
│   ├── main.py
│   ├── pysurv_model_training.py
│   ├── pysurv_validation.py
│   ├── pysurv_models.py
│   ├── environment.yml
├── scikit_surv
│   ├── a_submit.sh
│   ├── b_submit.sh
│   ├── main.py
│   ├── sksurv_model_training.py
│   ├── sksurv_validation.py
│   ├── sksurv_models.py
│   ├── environment.yml
└── utils
    ├── data_preprocessing.py
    ├── logger.py
    ├── utils.py
    ├── model_validation.py
```

## Directories Explained

- **config**: Holds configuration files like `column_names.py` (defines dataset column names) and `config.py` (contains project-wide parameters like paths and filenames).
- **pysurv**: Contains scripts for the PySurvival pipeline. Scripts include job submission scripts, main script, model training, validation, and model definitions.
- **scikit_surv**: Contains scripts for the scikit-survival pipeline. Scripts include job submission scripts, main script, model training, validation, and model definitions.
- **utils**: Contains utility scripts for data loading, preprocessing, logging, and other utilities.

## Setting Up The Environment

Each pipeline has its own `environment.yml` file as the package versions are not compatible across pipelines. You can use these `environment.yml` files to create a separate conda environment for each pipeline. Here is how to create a conda environment using the `environment.yml` file:

```bash
$ conda env create -f environment.yml
```

Then, activate the newly created environment:

```bash
$ conda activate myenv
```
Replace `myenv` with the name of your environment specified in the `environment.yml` file.


## Running the Pipelines

To execute the pipelines, navigate to either the `pysurv` or `scikit_surv` directory and run the main script (main.py). Ensure that your data is ready and that you've set the correct parameters in the config files.

For PySurvival pipeline:

```bash
$ cd pysurv
$ conda activate pysurv
$ python main.py
```

Replace `pysurv` with `scikit_surv` for the scikit-survival pipeline.

For cluster execution, use the provided shell scripts (`a_submit.sh` and `b_submit.sh`).

```bash
$ bash a_submit.sh
```

## Feature Importance

The pipeline can calculate permutation feature importance for PySurvival using the C-Index, which can be enabled/disabled via a parameter.

## Logging

The pipeline logs each run to a directory specified in the configuration.


### PySurv Pipeline

For the PySurvival pipeline, `ParamFactory` is used to create models and parameter grids.

```python
param_factory = ParamFactory(model="non_linear_cox", is_grid=False)
model, param_grid = param_factory.get_params()
```
For this pipline the grid seach can be triigered by setting `is_grid=True`.

### Scikit_surv Pipeline

For the scikit-survival pipeline, `SurvivalModelFactory` and `ModelParameterGridFactory` are used to create models and parameter grids.

```python
model = SurvivalModelFactory.get_coxnet_model()
param_grid = ModelParameterGridFactory.get_coxnet_param_grid()
concordance_wrappper = ScorerFactory.as_concordance_index_ipcw_scorer(model, y_train)
random_search = create_randomized_search(param_grid, concordance_wrapper)
```
You can either use the model, concordance wrapper, and random search objects in the training process depending on your needs.

```python
preprocess_train_validate(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    preprocessor=preprocessor,
    feature_selector=None,
    model_path=model_path,
    calculate_permutation_importance=True,
)
```
