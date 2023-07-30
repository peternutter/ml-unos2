import datetime
import logging
import time
import traceback

import pandas as pd
import numpy as np
import psutil


def calculate_tau(df):
    max_time = df[df["status"] == True]["time"].max()
    return max_time


def log_memory_periodically(interval=500):
    while True:
        try:
            log_memory()
            time.sleep(interval)
        except Exception as e:
            logging.error(f"Failed to log memory: {str(e)}")
            break

def log_memory():
    memory_stats = psutil.virtual_memory()
    available_memory = memory_stats.available
    total_memory = memory_stats.total
    used_memory = memory_stats.used
    percent_used = memory_stats.percent
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"{current_time} - Total memory: {total_memory / (1024 ** 2)} MB, \
                  Used memory: {used_memory / (1024 ** 2)} MB, \
                  Available memory: {available_memory / (1024 ** 2)} MB, \
                  Percent used: {percent_used}%")


def log_model_params(model):
    """
    Log model parameters and scores if they exist.

    Parameters:
    model (sklearn.base.BaseEstimator): The model to log parameters and scores from.
    """

    param_attrs = [
        "get_params",
        "train_score_",
        "oob_score_",
        "best_score_",
        "best_params_",
        "feature_importances_",
        "estimator.feature_importances_",
    ]

    for attr in param_attrs:
        try:
            value = getattr(model, attr)
            logging.info(f"{attr}: {value}")
        except (NotImplementedError, AttributeError):
            logging.info(f"{attr} not implemented or does not exist for this model.")

def log_error(error):
    logging.error(f"Error: {str(error)}")
    logging.error(traceback.format_exc())


