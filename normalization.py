import pandas as pd
import os
import re


def aggressive_text_denoise(text):
    """
    Aggressively clean and denoise text data to remove all artifacts and noise.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs and emails
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove timestamps, hashtags, mentions
    text = re.sub(r'\[\d{2}:\d{2}(?::\d{2})?\]', '', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    
    # Remove repeated characters (stuttering)
    text = re.sub(r'(\w)\1{2,}', r'\1', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Convert to lowercase and strip
    text = text.lower().strip()
    
    # Remove non-ASCII characters except basic punctuation
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    return text


def clean_and_normalize(file_path):
    if not os.path.exists(file_path):
        return

    df = pd.read_csv(file_path)

    # 1. Aggressively denoise transcripts
    if 'transcript' in df.columns:
        df['transcript'] = df['transcript'].fillna("").apply(aggressive_text_denoise).str.strip()

    # 2. Standardize and clean labels
    if 'label' in df.columns:
        df['label'] = df['label'].astype(str).str.lower().str.strip()

    # 3. Fix the 6% 'Lie' Class
    # Option A: Merge 'lie' into 'deception' (Recommended for higher accuracy)
    df['label'] = df['label'].replace('lie', 'deception')

    # 4. Remove rows with empty transcripts
    df = df[df['transcript'].str.len() > 0]
    
    # 5. Remove duplicates based on transcript and label
    df = df.drop_duplicates(subset=['transcript', 'label'], keep='first')

    df.to_csv(file_path, index=False)
    print(f"Normalized {file_path} | Rows: {len(df)}")


# Run for all your splits
for f in ["data/train_split.csv", "data/test_split.csv"]:
    clean_and_normalize(f)