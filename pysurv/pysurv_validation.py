

import logging
import traceback
import numpy as np
from scikit_surv.sksurv_validation import calculate_auc, calculate_c_index_surv, calculate_c_ipcws, mask_data
from sksurv.metrics import integrated_brier_score

from utils.utils import calculate_tau, log_error

def calculate_IBS(model, X_val, y_val, y_train):
    score_brier = -np.inf
    try:
        X_val_s, y_val_s = mask_data(X_val, y_val, y_train)
        times = [abs(a) for (a, b) in model.time_buckets]
        times.append(abs(model.time_buckets[-1][1]))
        times = np.array(times)
        low = np.percentile(y_train["time"], 10)
        high = np.percentile(y_train["time"], 90)
        mask = (times > low) & (times < high)
        times = times[mask]
        surv_prob = np.row_stack([fn for fn in model.predict_survival(X_val_s)])
        surv_prob = surv_prob[:, mask]
        score_brier = integrated_brier_score(y_train, y_val_s, surv_prob, times)
        logging.info(f"IBS: {score_brier}")
    except Exception as e:
        log_error(e)
    return score_brier



def validate_model(model, X_val, y_val, y_train=None):
    tau = calculate_tau(y_train)
    try:
        y_pred = model.predict_risk(X_val)
    except Exception as e:
        logging.error(f"Error in model prediction: {str(e)}")
        logging.error(traceback.format_exc())
        return -np.inf, -np.inf, -np.inf, -np.inf

    c_index_surv = calculate_c_index_surv(y_val, y_pred)
    ibs = calculate_IBS(model, X_val, y_val, y_train)
    c_ipcws = calculate_c_ipcws(y_train, y_val, y_pred, tau)
    mean_auc = calculate_auc(y_val,y_train, y_pred)

    return c_index_surv, c_ipcws, mean_auc, ibs