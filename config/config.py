import os

# Get the current file's directory
current_file_dir = os.path.dirname(os.path.abspath(__file__))

# Define the project root directory (assuming this script is in src/)
project_root_dir = os.path.join(current_file_dir, "..")

# Configuration settings
timestamp_format = "%m%d_%H%M"
output_dir_format = "output/{timestamp}"
model_path_format = "{output_dir}/model.joblib"
log_filename_format = "{output_dir}/model_training.log"
data_filename_feather = os.path.join(project_root_dir, "../data/data.feather")
