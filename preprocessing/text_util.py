import re
import torch

def clean_transcript(text):
    """
    Basic text cleaning for the Linguistic Brain.
    """
    if not text:
        return ""
    # Lowercase and remove special characters/punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
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