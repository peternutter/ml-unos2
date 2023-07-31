import logging
import threading
import time
import traceback

import numpy as np
from utils.data_preprocessing import preprocess_data
from utils.utils import calculate_tau, log_memory_periodically
from sksurv.util import Surv
from sksurv.metrics import *
from sklearn.model_selection import ParameterGrid
from pysurv.pysurv_validation import calculate_IBS, calculate_and_save_permutation_importance, validate_model

def preprocess_train_validate_pysurv(model, X_train, y_train, X_val, y_val, preprocessor, feature_selector, scorer, model_path, param_grid, calculate_feature_importance=False):
    X_train_selected = preprocess_data(X_train, y_train, preprocessor, feature_selector, fit=True)
    X_val_selected = preprocess_data(X_val, y_val, preprocessor, feature_selector, fit=False)

    log_memory_thread = threading.Thread(target=log_memory_periodically)
    log_memory_thread.daemon = True
    log_memory_thread.start()

    best_params, best_score, best_model = hyperparameter_tuning(model, X_train_selected, y_train, X_val_selected, y_val, param_grid, scorer=scorer)

    best_model.save(model_path)

    if hasattr(best_model, "structure"):
        logging.info(f"Model structure: {best_model.structure}")

    c_index_surv, c_ipcws, mean_auc, ibs = validate_model(best_model, X_val_selected, y_val, y_train)
    if calculate_feature_importance:
        importances = calculate_and_save_permutation_importance(best_model, X_val_selected, y_val, preprocessor, model_path)
    return best_model




def initialize_model(model, params):
    model_name = str(model.__name__)
    if model_name in ["NonLinearCoxPHModel", "NeuralMultiTaskModel"]:
        structure = params.pop("structure") 
        if model_name == "NeuralMultiTaskModel":
            bins = params.pop("bins")
            return model(structure=structure, bins=bins)
        else:
            return model(structure=structure)
    elif model_name in ["RandomSurvivalForestModel", "ConditionalSurvivalForestModel"]:
        return model(num_trees=params.pop("num_trees"))
    elif model_name in ["ExponentialModel", "WeibullModel", "LogNormalModel", "LogLogisticModel", "GompertzModel"]:
        return model(bins=params.pop("bins"))
    else:
        return model()

def calculate_score(scorer, survival_model, y_train, y_val, X_val):
    if scorer == concordance_index_ipcw:
        tau = calculate_tau(y_train)
        return concordance_index_ipcw(y_train, y_val, survival_model.predict_risk(X_val), tau)[0]
    elif scorer == integrated_brier_score:
        return -calculate_IBS(survival_model, y_train, X_val, y_val)

def fit_and_score( model, params, X_train, y_train, X_val, y_val, scorer = concordance_index_ipcw):
    survival_model = initialize_model(model, params)

    try:
        survival_model.fit(X_train, y_train["time"], y_train["status"], **params)
        score = calculate_score(scorer, survival_model, y_train, y_val, X_val)
    except (ValueError, Exception) as e:
        logging.error(f"Error: {e}")
        traceback.print_exc()
        return -np.inf, params, survival_model

    return score, params, survival_model

def hyperparameter_tuning(model, X_train, y_train, X_val, y_val, hyperparameters, scorer=concordance_index_ipcw):
    param_grid = ParameterGrid(hyperparameters)
    model_name = str(model.__name__)
    logging.info(f"Starting hyperparameter tuning {model_name} with {len(param_grid)} combinations")

    best_params = None
    best_score = -np.inf
    best_model = None

    for i, params in enumerate(param_grid, 1):
        logging.info(f"Round {i} of {len(param_grid)}")
        start_time = time.perf_counter()
        score, params, survival_model = fit_and_score(model, params, X_train, y_train, X_val, y_val, scorer)
        elapsed_time = time.perf_counter() - start_time
        logging.info(f"Score: {score}, Params: {params}, Time taken: {elapsed_time} seconds")
        if score > best_score:
            best_score = score
            best_params = params
            best_model = survival_model
            logging.info(f"New best score: {best_score}, Params: {best_params}")

    logging.info(f"Best params: {best_params}, Best score: {best_score}")
    return best_params, best_score, best_model
