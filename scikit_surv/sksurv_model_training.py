import threading
from scikit_surv.sksurv_validation import calculate_and_save_permutation_importance, validate_model
from utils.data_preprocessing import preprocess_data


import logging

from utils.utils import log_memory_periodically, log_model_params


def train_model(model, X_train, y_train, model_path):
    """
    Train a given model and save it to the provided path.

    Parameters:
    model (sklearn.base.BaseEstimator): The model to be trained.
    X_train (pd.DataFrame): The input data for training.
    y_train (pd.DataFrame): The labels for training.
    model_path (str): The path to save the trained model.

    Returns:
    model (sklearn.base.BaseEstimator): The trained model.
    """

    # Start a separate thread to log memory usage periodically
    log_memory_thread = threading.Thread(target=log_memory_periodically)
    log_memory_thread.daemon = True
    log_memory_thread.start()

    # Recursively log the model's class name
    model_temp = model
    while hasattr(model_temp, "estimator"):
        logging.info(f"Model: {model_temp.__class__.__name__}")
        model_temp = model_temp.estimator

    model.fit(X_train, y_train)
    log_memory_periodically.interval = float("inf")

    dump(model, model_path, compress=5)

    log_model_params(model)

    return model


def preprocess_train_validate(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        preprocessor,
        feature_selector,
        model_path,
        calculate_permutation_importance=False,
):
    """
 This function preprocesses training and validation data, trains a model, validates the model,
 and optionally calculates the permutation importance.

 Parameters:
 model (sklearn.BaseEstimator): The estimator object to be trained.
 X_train (pd.DataFrame): The training data.
 y_train (pd.DataFrame): The training labels.
 X_val (pd.DataFrame): The validation data.
 y_val (pd.DataFrame): The validation labels.
 preprocessor (sklearn.TransformerMixin): The preprocessor object (transformer).
 feature_selector (sklearn.FeatureSelection): The feature selector object.
 model_path (str): The path to save the trained model.
 calcualte_permutation_importance (bool, optional): If True, calculate and log the permutation importance.

 Returns:
 model (sklearn.BaseEstimator): The trained model.

 """
    logging.info("Starting data preprocessing")
    X_train_selected = preprocess_data(
        X_train, y_train, preprocessor, feature_selector, fit=True
    )
    X_val_selected = preprocess_data(
        X_val, y_val, preprocessor, feature_selector, fit=False
    )

    logging.info("Starting model training")
    model = train_model(
        model=model, X_train=X_train_selected, y_train=y_train, model_path=model_path
    )

    logging.info("Starting model validation")
    c_index_surv, c_ipcws, mean_auc, integrated_brier_score = validate_model(
        model=model, X_val=X_val_selected, y_val=y_val, y_train=y_train
    )
    if calculate_permutation_importance:
        calculate_and_save_permutation_importance(model, X_val_selected, y_val, preprocessor, model_path)

    return model