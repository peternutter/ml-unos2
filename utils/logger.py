import argparse
import logging
import os

from config.config import output_dir_format, log_filename_format

# Include this line to prevent potential multiprocessing issues with NumPy in certain cases
os.environ["PYTHONHASHSEED"] = "0"


def parse_arguments():
    """
    Parses command-line options.

    Returns:
        Namespace: The parsed arguments as a Namespace object.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Timestamp for log and output directory",
    )
    return parser.parse_args()


def get_output_dir(timestamp):
    """
    Generates the output directory based on a given timestamp.

    Args:
        timestamp (str): The given timestamp.

    Returns:
        str: The output directory path.
    """
    output_dir = output_dir_format.format(timestamp=timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def setup_logging(output_dir):
    """
    Sets up the logging configuration.

    Args:
        output_dir (str): The output directory to save logs.
    """
    logging.basicConfig(
        filename=log_filename_format.format(output_dir=output_dir),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info(f"Saving model and logs to {output_dir}")
