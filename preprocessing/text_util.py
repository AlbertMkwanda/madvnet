import re
import torch

def clean_transcript(text):
    """
    Aggressive text cleaning for the Linguistic Brain.
    Removes noise, artifacts, and normalizes text while preserving semantic content.
    """
    if not text:
        return ""
    
    text = str(text)
    
    # Remove URLs and emails
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove timestamps, hashtags, mentions
    text = re.sub(r'\[\d{2}:\d{2}(?::\d{2})?\]', '', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    
    # Remove repeated characters (stuttering)
    text = re.sub(r'(\w)\1{2,}', r'\1', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Lowercase
    text = text.lower()
    
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Remove special characters except basic punctuation (keep for semantic meaning)
    text = re.sub(r'[^a-z0-9\s.!?,\'-]', '', text)
    
    return text.strip()

def tokenize_for_model(text, tokenizer, max_length=128):
    """
    Converts raw text into input IDs and attention masks for BERT-based models.
    """
    cleaned = clean_transcript(text)
    encoded = tokenizer.encode_plus(
        cleaned,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return encoded['input_ids'], encoded['attention_mask']