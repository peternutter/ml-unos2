import os
import sys
from datetime import datetime

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from utils.logger import get_output_dir, setup_logging, parse_arguments
from utils.data_preprocessing import load_data, split_data, create_preprocessor
import logging
import numpy as np
from sklearn.decomposition import PCA
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc
from config.config import timestamp_format, data_filename_feather, model_path_format
from config.column_names import BASIC_COLUMNS, EXTRA_COLUMNS, OUTPUT_VARS
from scikit_surv.sksurv_models import SurvivalModelFactory, ModelParameterGridFactory, ScorerFactory

def main():
    args = parse_arguments()

    # If the timestamp is not provided, use the current time
    timestamp = args.timestamp if args.timestamp else datetime.now().strftime(timestamp_format)

    output_dir = get_output_dir(timestamp)
    setup_logging(output_dir)
    data_path = data_filename_feather
    columns_numeric = BASIC_COLUMNS["numeric"]
    columns_categorical = BASIC_COLUMNS["categorical"]
    indicator_columns = OUTPUT_VARS["kidney"]
    df = load_data(data_path, columns_numeric, columns_categorical, indicator_columns, sample=0.01)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, indicator_columns, test_size=0.2, val_size=0.2, transform_y=True
    )

    preprocessor = create_preprocessor(columns_numeric, columns_categorical, min_frequency=None)


    # Train and save model
    logging.info("Training and saving model...")

    model_path = model_path_format.format(output_dir=output_dir)
    model = SurvivalModelFactory.get_coxnet_model()
    concordance_wrappper = ScorerFactory.as_concordance_index_ipcw_scorer(model, tau=tau)

    concordance_wrappper = as_concordance_index_ipcw_scorer(model, tau=tau)
    param_grid = get_gbsa_param_grid()
    # # grid_search = create_grid_search(param_grid, tau=tau, model=model_wrapped, n_jobs=8)
    random_search = create_randomized_search(
        param_distributions=param_grid,
        tau=tau,
        model=concordance_wrappper,
        n_jobs=8,
        n_iter=800,
        n_splits=5,
    )

    # Feature selection
    # k_best = 30
    # feature_selector = get_k_best_selector(k_best)
    pca = PCA(n_components=0.99, random_state=42)

    preprocess_train_validate(
        model=concordance_wrappper,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        preprocessor=preprocessor,
        feature_selector=None,
        tau=tau,
        model_path=model_path,
        calcualte_permutation_importance=True,
    )


if __name__ == "__main__":
    main()
