import pandas as pd
import os


def clean_and_normalize(file_path):
    if not os.path.exists(file_path):
        return

    df = pd.read_csv(file_path)

    # 1. Standardize Text
    for col in ['transcript', 'label']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()

    # 2. Fix the 6% 'Lie' Class
    # Option A: Merge 'lie' into 'deception' (Recommended for higher accuracy)
    df['label'] = df['label'].replace('lie', 'deception')

    # Option B: Keep them separate but clean strings
    # (Already handled by .str.lower())

    df.to_csv(file_path, index=False)
    print(f"Normalized {file_path}")


# Run for all your splits
for f in ["data/train_split.csv", "data/test_split.csv"]:
    clean_and_normalize(f)