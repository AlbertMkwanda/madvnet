import pandas as pd
import os
import re


def denoise_text_content(text):
    """Remove noise and artifacts from text while preserving semantic content."""
    if not isinstance(text, str):
        return ""
    
    # Remove URLs and emails
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove timestamps, hashtags, mentions
    text = re.sub(r'\[\d{2}:\d{2}(?::\d{2})?\]', '', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    
    # Remove repeated characters
    text = re.sub(r'(\w)\1{2,}', r'\1', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Clean special characters
    text = text.lower().strip()
    
    return text


def normalize_csv(file_path):
    if not os.path.exists(file_path):
        print(f"Skipping: {file_path} (File not found)")
        return

    # Load the data
    df = pd.read_csv(file_path)

    # 1. Denoise and normalize transcripts
    if 'transcript' in df.columns:
        df['transcript'] = df['transcript'].fillna("").apply(denoise_text_content).str.strip()

    # 2. Lowercase the Labels
    if 'label' in df.columns:
        df['label'] = df['label'].fillna("").str.lower().str.strip()

    # 3. Remove rows with empty transcripts
    df = df[df['transcript'].str.len() > 0]

    # Save the cleaned version back to the same path
    df.to_csv(file_path, index=False)
    print(f"Successfully cleaned: {file_path} | Final rows: {len(df)}")


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