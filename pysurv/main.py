
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import logging
from xmlrpc.client import DateTime
from pysurv.pysurv_model_training import preprocess_train_validate_pysurv
from utils.data_preprocessing import create_preprocessor, load_data, split_data
from utils.logger import get_output_dir, parse_arguments, setup_logging
from config.config import timestamp_format, data_filename_feather, model_path_format
from config.column_names import BASIC_COLUMNS, OUTPUT_VARS
from pysurv.pysurv_models import ParamFactory
from sklearn.decomposition import PCA
from sksurv.metrics import concordance_index_ipcw


def main():
    args = parse_arguments()

    # If the timestamp is not provided, use the current time
    timestamp = (
        args.timestamp if args.timestamp else DateTime.now().strftime(timestamp_format)
    )

    output_dir = get_output_dir(timestamp)
    setup_logging(output_dir)
    data_path = data_filename_feather
    columns_numeric = BASIC_COLUMNS["numeric"]
    columns_categorical = BASIC_COLUMNS["categorical"]
    indicator_columns = OUTPUT_VARS["kidney"]
    df = load_data(
        data_path, columns_numeric, columns_categorical, indicator_columns, sample=0.5
    )

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, test_size=0.2, val_size=0.2, transform_y=True
    )

    preprocessor = create_preprocessor(
        columns_numeric, columns_categorical, min_frequency=None
    )

    model_path = model_path_format.format(output_dir=output_dir)

    param_factory = ParamFactory(model="coxPH", is_grid=True)
    model, param_grid = param_factory.get_params()
    pca = PCA(n_components=0.99)
    # Train and save model
    logging.info("Training and saving model...")
    preprocess_train_validate_pysurv(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        preprocessor=preprocessor,
        feature_selector=pca,
        scorer=concordance_index_ipcw,
        model_path=model_path,
        param_grid=param_grid,
    )


if __name__ == "__main__":
    main()
