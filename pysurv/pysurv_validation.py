import logging
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sksurv.metrics import (
    integrated_brier_score,
    concordance_index_censored,
)

from utils.model_validation import (
    calculate_auc,
)
from utils.model_validation import calculate_c_index_surv, calculate_c_ipcws, mask_data
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
    mean_auc = calculate_auc(y_val, y_train, y_pred)

    return c_index_surv, c_ipcws, mean_auc, ibs


def calculate_and_save_permutation_importance(
        model, X_test, y_test, preprocessor, model_path, n_repeats=15, n_jobs=-1
):
    try:
        logging.info("Calculating permutation importance...")
        feature_importances_mean, feature_importances_std = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=n_repeats,
            n_jobs=n_jobs,
        )
        feature_names = get_feature_names(preprocessor)
        df_importances = pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean": feature_importances_mean,
                "importance_std": feature_importances_std,
            }
        )
        df_importances = df_importances.sort_values(by='importance_mean', ascending=False)
        logging.info(f"Permutation importance: {df_importances}")
        # Convert string to path
        path = Path(model_path)
        df_importances.to_csv(path.parent / "permutation_importance.csv")
    except Exception as e:
        log_error(e)
        df_importances = pd.DataFrame()
    return df_importances


def calculate_importance(j, model, X, y, baseline, n_repeats):
    importances_j = np.zeros(n_repeats)
    for i in range(n_repeats):
        X_temp = np.copy(X)  # Create a copy of the original data
        X_temp[:, j] = np.random.rand(X_temp[:, j].shape[0])
        y_perm = model.predict_risk(X_temp)  # Compute predictions on the permuted data
        m = concordance_index_censored(y["status"], y["time"], y_perm)[
            0
        ]  # Compute the metric on the permuted data
        importances_j[i] = baseline - m

    return importances_j


def permutation_importance(model, X, y, n_repeats=100, n_jobs=1):
    """
    model : PyTorch model
        Model object that you can call `.predict` on.
    X : np.ndarray
        Validation set matrix with shape `(n_samples, n_features)`.
    y : np.ndarray
        Target relative to `X`.
    metric : Callable
        Metric function applied on the validation set.
    n_repeats : int, optional (default=100)
        Number of times to permute each feature and compute the feature importance.
    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
    """
    y_pred = model.predict_risk(X)  # Compute the initial predictions
    baseline = concordance_index_censored(y["status"], y["time"], y_pred)[0]
    print(baseline)  # Compute the baseline metric

    feature_importances = Parallel(n_jobs=n_jobs)(
        delayed(calculate_importance)(j, model, X, y, baseline, n_repeats)
        for j in range(X.shape[1])
    )

    feature_importances = np.array(feature_importances)
    feature_importances_mean = np.mean(feature_importances, axis=1)
    feature_importances_std = np.std(feature_importances, axis=1)

    return feature_importances_mean, feature_importances_std


def get_feature_names(column_transformer):
    col_name = []
    for (
            transformer_in_columns
    ) in (
            column_transformer.transformers_
    ):  # the last transformer is ColumnTransformer's 'remainder'
        raw_col_name = transformer_in_columns[2]
        if isinstance(transformer_in_columns[1], Pipeline):
            transformer = transformer_in_columns[1].steps[-1][1]
        else:
            transformer = transformer_in_columns[1]
        try:
            names = transformer.get_feature_names_out(raw_col_name)
        except (
                AttributeError
        ):  # if no 'get_feature_names_out' function, use raw column name
            names = raw_col_name
        if isinstance(names, np.ndarray):  # eg.
            col_name += names.tolist()
        elif isinstance(names, list):
            col_name += names
        elif isinstance(names, str):
            col_name.append(names)
    return col_name
