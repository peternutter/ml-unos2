import numpy as np
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc


import logging
import traceback

from utils.utils import calculate_tau


def calculate_c_index_surv(y_val, y_pred):
    try:
        c_index_surv = concordance_index_censored(y_val["status"], y_val["time"], y_pred)
        logging.info(f"C_index: {c_index_surv}")
        return c_index_surv
    except Exception as e:
        logging.error(f"Error in calculating c_index_surv: {str(e)}")
        logging.error(traceback.format_exc())
        return -np.inf


def mask_data(X_val, y_val, y_train):
    tau = calculate_tau(y_train)
    mask = y_val["time"] < tau
    y_val_s = y_val[mask]
    X_val_s = X_val[mask]
    return X_val_s, y_val_s


def calculate_c_ipcws(y_train, y_val, y_pred, tau):
    try:
        c_ipcws = concordance_index_ipcw(y_train, y_val, y_pred, tau=tau)
        logging.info(f"Concordance index ipcw: {c_ipcws}")
        return c_ipcws
    except Exception as e:
        logging.error(f"Error in calculating c_ipcws: {str(e)}")
        logging.error(traceback.format_exc())
        return -np.inf


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