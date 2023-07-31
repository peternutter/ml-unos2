import logging
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sksurv.metrics import *
from utils.utils import calculate_tau, log_error


def validate_model(model, X_val, y_val, y_train=None):
    tau = calculate_tau(y_train)
    try:
        y_pred = model.predict(X_val)
    except Exception as e:
        logging.error(f"Error in model prediction: {str(e)}")
        logging.error(traceback.format_exc())
        return -np.inf, -np.inf, -np.inf, -np.inf

    c_index_surv = calculate_c_index_surv(y_val, y_pred)
    ibs = calculate_IBS(model, X_val, y_val, y_train)
    c_ipcws = calculate_c_ipcws(y_train, y_val, y_pred, tau)
    mean_auc = calculate_auc(y_val,y_train, y_pred)

    return c_index_surv, c_ipcws, mean_auc, ibs


def calculate_c_index_surv(y_val, y_pred):
    try:
        c_index_surv = concordance_index_censored(y_val["status"], y_val["time"], y_pred)
        logging.info(f"C_index: {c_index_surv}")
        return c_index_surv
    except Exception as e:
        logging.error(f"Error in calculating c_index_surv: {str(e)}")
        logging.error(traceback.format_exc())
        return -np.inf


def calculate_c_ipcws(y_train, y_val, y_pred, tau):
    try:
        c_ipcws = concordance_index_ipcw(y_train, y_val, y_pred, tau=tau)
        logging.info(f"Concordance index ipcw: {c_ipcws}")
        return c_ipcws
    except Exception as e:
        logging.error(f"Error in calculating c_ipcws: {str(e)}")
        logging.error(traceback.format_exc())
        return -np.inf


def mask_data(X_val, y_val, y_train):
    tau = calculate_tau(y_train)
    mask = y_val["time"] < tau
    y_val_s = y_val[mask]
    X_val_s = X_val[mask]
    return X_val_s, y_val_s

def calculate_IBS(model, X_val, y_val, y_train):
    score_brier = -np.inf
    try:
        X_val_s, y_val_s = mask_data(X_val, y_val, y_train)
        times = np.percentile(y_val_s["time"], np.linspace(10, 90, 100))
        estimator = model
        if hasattr(model, "best_estimator_"):
            estimator = model.best_estimator_
        surv_prob = np.row_stack(
            [fn(times) for fn in estimator.predict_survival_function(X_val_s)]
        )
        score_brier = integrated_brier_score(y_train, y_val_s, surv_prob, times)
        logging.info(f"IBS: {score_brier}")
    except Exception as e:
        logging.error(f"Error in calculate_IBS: {str(e)}")
        logging.error(traceback.format_exc())
    return score_brier

def calculate_auc(y_val, y_train, risk):
    mean_auc = -np.inf
    try:
        tau = calculate_tau(y_train)
        mask = y_val["time"] < tau
        y_val_s = y_val[mask]
        risk_s = risk[mask]
        times = np.percentile(y_train["time"], np.linspace(10, 90, 100))
        times = times[times < tau]
        auc, mean_auc = cumulative_dynamic_auc(y_train, y_val_s, risk_s, times=times)
        logging.info(f"Mean AUC: {mean_auc} using risk")
    except Exception as e:
        logging.error(f"Error in calculate_auc: {str(e)} using risk")
        logging.error(traceback.format_exc())
    return mean_auc


def calculate_and_save_permutation_importance(
        model, X_test, y_test, preprocessor, model_path, n_repeats=15, n_jobs=4):
    logging.info("Calculating permutation importance...")
    try:
        result = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=n_jobs,
        )
        feature_names = preprocessor.get_feature_names_out()
        print(feature_names)
        importance_df = build_importance_df(result, feature_names)
        logging.info(f"Permutation importance: {importance_df}")
        # Convert string to path
        path = Path(model_path)
        importance_df.to_csv(path.parent / "permutation_importance.csv")
    except Exception as e:
        log_error(e)
        importance_df = pd.DataFrame()
    return importance_df


def build_importance_df(result, feature_names):
    return pd.DataFrame({
        "importances_mean": result["importances_mean"],
        "importances_std": result["importances_std"],
    },
        index=feature_names,
    ).sort_values(by="importances_mean", ascending=False)