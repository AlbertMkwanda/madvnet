# config.py
import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
IS_COLAB = os.path.exists('/content/drive')

if IS_COLAB:
    # Use Drive paths
    ROOT_DIR = '/content/drive/MyDrive/Projects/main-project/'
else:
    # Use your local Windows paths
    ROOT_DIR = 'C:/Users/User/Documents/Projects/main-project/'

# Now build paths dynamically
DATA_DIR = os.path.join(ROOT_DIR, "data")
CLIPS_DIR = os.path.join(DATA_DIR, "clips/")
ORIGINAL_CSV = os.path.join(DATA_DIR, "dolos_processed.csv")
# Dataset Paths

FINAL_CSV = os.path.join(DATA_DIR, "dolos_final_training_retranscribed_medium_dedup_complete.csv")
SEGMENTED_DATASET_CSV=os.path.join(DATA_DIR, "segmented_dataset.csv")
SEGMENTED_CLIPS_DIR = os.path.join(DATA_DIR, "segmented_clips/")
SEGMENTED_CLIPS_DIR=os.path.join(DATA_DIR, "segmented_clips/")
# Model Parameters
IMG_SIZE = 112  # Standard for 3D-CNN/Vision Transformers
SEQUENCE_LENGTH = 16  # Number of frames per video clip
BATCH_SIZE = 8  # Keep low for i7-4600U RAM limits
DEVICE = "CPU"  # Intel HD Graphics Family optimization

DATASET_PATH = os.path.join(DATA_DIR, "")