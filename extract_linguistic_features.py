import torch
import numpy as np
import pandas as pd
import spacy
import os
import re
from textblob import TextBlob
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
import config

# ==========================================
# 1. SETUP & MODEL LOADING
# ==========================================
# Load spaCy for POS tagging and linguistic analysis
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load RoBERTa-base (768 hidden units)
print(f"Loading RoBERTa-base on {device}...")
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base').to(device)
model.eval()


# ==========================================
# 2. TEXT DENOISING LOGIC
# ==========================================
def denoise_text(text):
    """
    Aggressively denoise and normalize text by removing artifacts,
    extra whitespace, and non-linguistic noise while preserving semantic content.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""
    
    # 1. Remove URLs and email addresses
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    
    # 2. Remove timestamps and special formatting (e.g., [00:23], **, ##)
    text = re.sub(r'\[\d{2}:\d{2}(?::\d{2})?\]', '', text)
    text = re.sub(r'[\*_]{2,}', '', text)
    text = re.sub(r'#{1,6}\s+', '', text)
    
    # 3. Remove repeated characters (stuttering noise like "soooooo" -> "so")
    text = re.sub(r'(\w)\\1{2,}', r'\1', text)
    
    # 4. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # 5. Convert to lowercase
    text = text.lower()
    
    # 6. Remove special characters except basic punctuation
    text = re.sub(r'[^a-z0-9\s.!?,\'-]', '', text)
    
    # 7. Fix spacing around punctuation
    text = re.sub(r'\s+([.!?,])', r'\1', text)
    text = re.sub(r"(['])(\s)", r'\1', text)
    
    return text


# ==========================================
# 3. FEATURE EXTRACTION LOGIC
# ==========================================
def get_liu_handcrafted_features(text):
    """Extracts the 7 Theoretical Deception Cues based on Liu et al."""
    doc = nlp(text)
    blob = TextBlob(text)

    # 1. Psychological Distancing (1st Person Pronouns)
    i_pronouns = sum(1 for t in doc if t.text.lower() in ['i', 'me', 'my', 'mine'])

    # 2. Narrativity (Verb/Noun ratio)
    nouns = sum(1 for t in doc if t.pos_ == "NOUN")
    verbs = sum(1 for t in doc if t.pos_ == "VERB")
    narrativity_ratio = verbs / (nouns + 1)

    # 3. Sentiment Tone (Polarity)
    sentiment_polarity = blob.sentiment.polarity

    # 4. Subjectivity
    sentiment_subjectivity = blob.sentiment.subjectivity

    # 5. Cognitive Complexity (Exclusive words)
    exclusive_words = sum(1 for t in doc if t.text.lower() in ['but', 'except', 'without', 'however'])

    # 6. Nonfluencies (Interjections/Fillers like um, uh)
    fillers = sum(1 for t in doc if t.pos_ == "INTJ")

    # 7. Lexical Diversity (Type-Token Ratio)
    tokens = [t.text.lower() for t in doc]
    ttr = len(set(tokens)) / (len(tokens) + 1)

    return [i_pronouns, narrativity_ratio, sentiment_polarity,
            sentiment_subjectivity, exclusive_words, fillers, ttr]


def extract_775_linguistic_features(csv_path, output_name):
    """Processes CSV to save 775-dim (768 RoBERTa + 7 Handcrafted) vectors"""
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    combined_features = []
    labels = []

    print(f"\n--- Extracting 775-dim Features for {output_name} ---")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row['transcript']) if pd.notnull(row['transcript']) else ""
        
        # Apply aggressive text denoising
        text = denoise_text(text)
        
        if len(text) == 0:
            continue

        # A. Neural Extraction (RoBERTa-base: 768-dim)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # full 768 dimensions
            roberta_vec = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

        # B. Handcrafted Extraction (7-dim)
        liu_vec = np.array(get_liu_handcrafted_features(text))

        # C. Concatenation (768 + 7 = 775)
        final_vec = np.concatenate([roberta_vec, liu_vec])
        combined_features.append(final_vec)

        # Handle labels (assuming column name is 'label')
        label = 1 if str(row['label']).lower() == 'deception' else 0
        labels.append(label)

    # Ensure output directory exists
    if not os.path.exists('data'):
        os.makedirs('data')

    # Save as .npy
    np.save(f"data/linguistic_{output_name}_x.npy", np.array(combined_features))
    np.save(f"data/linguistic_{output_name}_y.npy", np.array(labels))

    print(f"Saved: data/linguistic_{output_name}_x.npy | Shape: {np.array(combined_features).shape}")


# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    DATA_DIR = config.DATA_DIR
    
    print("\n" + "="*70)
    print("EXTRACTING LINGUISTIC FEATURES FOR ALL SPLITS")
    print("="*70)
    
    # Process all 4 splits
    extract_775_linguistic_features(DATA_DIR + "train_split.csv", "train")
    extract_775_linguistic_features(DATA_DIR + "rldd_test_split.csv", "test_rldd")
    extract_775_linguistic_features(DATA_DIR + "dolos_test_split.csv", "test_dolos")
    extract_775_linguistic_features(DATA_DIR + "dolos_validation_split.csv", "val_dolos")
    
    print("\n" + "="*70)
    print("✓ LINGUISTIC EXTRACTION COMPLETE")
    print("="*70)