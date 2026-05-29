import pandas as pd
import os


def normalize_csv(file_path):
    if not os.path.exists(file_path):
        print(f"Skipping: {file_path} (File not found)")
        return

    # Load the data
    df = pd.read_csv(file_path)

    # 1. Lowercase the Transcripts
    if 'transcript' in df.columns:
        # fillna("") prevents errors if there are empty transcripts
        df['transcript'] = df['transcript'].fillna("").str.lower().str.strip()

    # 2. Lowercase the Labels
    if 'label' in df.columns:
        df['label'] = df['label'].fillna("").str.lower().str.strip()



    # Save the cleaned version back to the same path
    df.to_csv(file_path, index=False)
    print(f"Successfully normalized: {file_path}")


if __name__ == "__main__":
    # List all the files that need normalization
    files_to_fix = [
         # Update this to your config.FINAL_CSV path
        "data/train_split.csv",
        "data/test_split.csv"
    ]

    print("--- Starting Dataset Normalization ---")
    for csv_file in files_to_fix:
        normalize_csv(csv_file)
    print("--- Normalization Complete ---")