import os
import sys
from datetime import datetime
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from utils.model_training import create_randomized_search
from utils.logger import get_output_dir, setup_logging, parse_arguments
from utils.data_preprocessing import load_data, split_data, create_preprocessor
import logging
import numpy as np
from sklearn.decomposition import PCA
from config.config import timestamp_format, data_filename_feather, model_path_format
from config.column_names import BASIC_COLUMNS, OUTPUT_VARS
from scikit_surv.sksurv_models import SurvivalModelFactory, ModelParameterGridFactory, ScorerFactory
from utils.model_training import preprocess_train_validate

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
    df = load_data(data_path, columns_numeric, columns_categorical, indicator_columns, sample=1)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, test_size=0.2, val_size=0.2, transform_y=True
    )

    preprocessor = create_preprocessor(columns_numeric, columns_categorical, min_frequency=None)

    # Train and save model
    logging.info("Training and saving model...")

    model_path = model_path_format.format(output_dir=output_dir)
    model = SurvivalModelFactory.get_random_forest_model()
    concordance_wrappper = ScorerFactory.as_concordance_index_ipcw_scorer(model, y_train)
    param_grid = ModelParameterGridFactory.get_coxnet_param_grid()
    random_search = create_randomized_search(param_grid, concordance_wrappper)
    # Feature selection
    pca = PCA(n_components=0.99, random_state=42)

    preprocess_train_validate(
        model=concordance_wrappper,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        preprocessor=preprocessor,
        feature_selector=None,
        model_path=model_path,
        calculate_permutation_importance=True,
    )


if __name__ == "__main__":
    main()
