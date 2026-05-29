# config.py
import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Dataset Paths
CLIPS_DIR = "C:/Users/User/Documents/Projects/main-project/data/clips/"
ORIGINAL_CSV ="C:/Users/User/Documents/Projects/main-project/data/dolos_processed.csv"
FINAL_CSV = "C:/Users/User/Documents/Projects/main-project/data/dolos_final_training_retranscribed_medium_dedup_complete.csv"

# Model Parameters
IMG_SIZE = 112  # Standard for 3D-CNN/Vision Transformers
SEQUENCE_LENGTH = 16  # Number of frames per video clip
BATCH_SIZE = 8  # Keep low for i7-4600U RAM limits
DEVICE = "CPU"  # Intel HD Graphics Family optimization

DATASET_PATH = "C:/Users/User/Documents/Projects/main-project/data/"