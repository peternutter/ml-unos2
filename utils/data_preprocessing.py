import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sksurv.util import Surv


def load_data(
        path,
        columns_numeric,
        columns_categorical,
        indicator_columns,
        dropna=False,
        sample=1.0,
        random_state=42,
):
    """
    Loads a data frame from a feather file, sets the index, filters columns,
    drops missing values (optional), and samples the data (optional).

    Args:
        path (str): Path to the feather file.
        columns_numeric (list): List of numeric column names.
        columns_categorical (list): List of categorical column names.
        indicator_columns (list): List of indicator column names.
        dropna (bool): Whether to drop rows with missing values. Default is False.
        sample (float): Fraction of rows to sample from the data. Default is 1.0.
        random_state (int): Seed for the random number generator. Default is 42.

    Returns:
        pandas.DataFrame: The loaded and preprocessed data frame.
    """
    # Log the load operation
    logging.info(
        f"Loading data from {path} with dropna={dropna} and sample={sample}"
    )

    # Load the data frame from the feather file
    df = pd.read_feather(path)

    # Set the index, filter columns, drop missing values (optional), and sample rows (optional)
    df.set_index("_id", inplace=True)
    df = df[columns_numeric + columns_categorical + indicator_columns]
    logging.info(f"Data loaded. Shape: {df.shape}")
    df = df.rename(columns={indicator_columns[0]: "status", indicator_columns[1]: "time"})
    if dropna:
        df.dropna(inplace=True)
        logging.info(f"Data after dropping NaNs. Shape: {df.shape}")

    df = df.sample(frac=sample, random_state=random_state)
    logging.info(f"Data after sampling. Shape: {df.shape}")

    return df


def split_data(df, test_size=0.2, val_size=0.2, transform_y=True, random_state=42):
    """
    Splits a dataframe into training, validation, and test sets.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to be split.
    test_size : float, optional
        The proportion of the dataset to include in the test split.
    val_size : float, optional
        The proportion of the dataset to include in the validation split.
    transform_y : bool, optional
        If True, transform the target variable using the `Surv.from_dataframe` method.
    random_state : int, optional
        The seed used by the random number generator for shuffling the data.

    Returns
    -------
    tuple of pandas.DataFrame
        The train, validation, and test splits of X, and the corresponding splits of y.
    """

    # Identify features and target variable
    indicator_columns = ["status", "time"]
    feature_cols = df.columns.drop(indicator_columns)
    X = df[feature_cols]

    if transform_y:
        y = Surv.from_dataframe("status", "time", df[indicator_columns])
    else:
        y = df[indicator_columns]

    logging.info(f"Shape of X: {X.shape}")

    # First, split the data into a (train + validation) set and a test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Then, split the (train + validation) set into a training set and a validation set
    # The size of the validation set will be val_size/(1-test_size) of the (train + validation) set
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size / (1 - test_size), random_state=random_state
    )

    logging.info(f"Split completed. Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_preprocessor(columns_numeric, columns_categorical, min_frequency=None):
    """
    Creates a preprocessing pipeline for numeric and categorical data.

    Parameters
    ----------
    columns_numeric : list
        A list of numeric column names.
    columns_categorical : list
        A list of categorical column names.
    min_frequency : int, optional
        The minimum frequency for a category to be considered (to include in encoding).

    Returns
    -------
    preprocessor : sklearn.compose.ColumnTransformer
        A ColumnTransformer that preprocesses numeric and categorical columns.
    """

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    encoder_params = {
        "drop": "if_binary",
        "handle_unknown": "ignore",
    }

    if min_frequency is not None:
        encoder_params["min_frequency"] = min_frequency

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(**encoder_params)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, columns_numeric),
        ("cat", categorical_transformer, columns_categorical),
    ])

    return preprocessor


def preprocess_data(X, y, preprocessor, feature_selector=None, fit=True):
    """
    Preprocess data using a specified preprocessor and optionally a feature selector.

    Parameters:
    X (pd.DataFrame): The input data.
    y (pd.DataFrame): The input labels.
    preprocessor (sklearn.TransformerMixin): The preprocessor object.
    feature_selector (sklearn.FeatureSelection, optional): The feature selector object.
    fit (bool, optional): If True, fit the preprocessor and (optionally) the feature selector on the data.

    Returns:
    X_selected (pd.DataFrame): The preprocessed and optionally feature-selected data.
    """

    # Fit or transform data with preprocessor
    X_transformed = preprocessor.fit_transform(X) if fit else preprocessor.transform(X)

    # Convert to dense if it's a sparse matrix
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    if feature_selector:
        # Fit or transform data with feature_selector
        X_selected = feature_selector.fit_transform(X_transformed, y) if fit else feature_selector.transform(
            X_transformed)

        # Log feature selection info if fitting
        if fit:
            if hasattr(feature_selector, "get_support"):
                logging.info(f"Selected features: {feature_selector.get_support(indices=True)}")
            if hasattr(feature_selector, "n_components_"):
                logging.info(f"Number of components: {feature_selector.n_components_}")
    else:
        X_selected = X_transformed

    logging.info(f"Preprocessed data shape: {X_selected.shape}")
    return X_selected


def create_randomized_search(
        param_distributions: dict,
        model=None,
        n_iter: int = 10,
        n_splits: int = 3,
        n_jobs: int = 2,
) -> RandomizedSearchCV:
    """
    Create a RandomizedSearchCV object for survival analysis.

    Args:
    param_distributions: Dictionary with parameters names (str) as keys
        and distributions or lists of parameters to try.
    model: The model to use for survival analysis. If None, RandomSurvivalForest will be used.
    n_iter: Number of parameter settings that are sampled.
    n_splits: Number of folds for cross-validation.
    n_jobs: Number of jobs to run in parallel.

    Returns:
    RandomizedSearchCV object.
    """
    logging.info(f"Randomized search, n_jobs={n_jobs}")

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    if hasattr(model, "estimator"):
        param_distributions = {
            f"estimator__{key}": value for key, value in param_distributions.items()
        }

    randomized_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        verbose=4,
        n_jobs=n_jobs,
        random_state=42,
        error_score=-np.inf,
        refit=True,
    )
    return randomized_search
