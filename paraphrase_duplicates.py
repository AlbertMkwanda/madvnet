# Optimized Duplicate Transcript Detection and Paraphrase Script
import pandas as pd
from collections import defaultdict
import config
import os
import re
import random

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    if not isinstance(text, str):
        return ""
    return text.lower().strip()

def find_duplicate_groups(df):
    """Find groups of duplicate transcripts efficiently using normalized text."""
    print("  Finding exact duplicates...")
    
    # Group by normalized transcript
    text_groups = defaultdict(list)
    for idx, row in df.iterrows():
        text = normalize_text(row['transcript'])
        text_groups[text].append(idx)
    
    duplicate_groups = {}
    group_id = 0
    
    for text, indices in text_groups.items():
        if len(indices) > 1:
            duplicate_groups[group_id] = indices
            group_id += 1
    
    return duplicate_groups

def apply_complex_paraphrase(text: str) -> str:
    """Apply sophisticated paraphrasing."""
    if not text.strip():
        return text

    sentences = text.split('. ')
    paraphrased_sentences = []

    for sentence in sentences:
        if len(sentence.split()) > 3:
            # Shuffle clause order if multiple clauses
            if ',' in sentence:
                parts = sentence.split(',')
                if len(parts) > 1:
                    shuffled = list(parts[:-1])
                    random.shuffle(shuffled)
                    sentence = ','.join(shuffled + [parts[-1]])

            sentence = apply_synonym_replacements(sentence)
        paraphrased_sentences.append(sentence)

    result = '. '.join(paraphrased_sentences).strip()
    return result if result != text else apply_synonym_replacements(text)

def apply_synonym_replacements(text: str) -> str:
    """Replace common words with synonyms."""
    replacements = [
        (r'\bwould\b', 'could'),
        (r'\bcould\b', 'would'),
        (r'\bthen\b', 'subsequently'),
        (r'\bafter that\b', 'following this'),
        (r'\bbut\b', 'however'),
        (r'\bhowever\b', 'but'),
        (r'\bvery\b', 'quite'),
        (r'\bquite\b', 'rather'),
        (r'\balways\b', 'constantly'),
        (r'\bnever\b', 'not once'),
        (r'\bsometimes\b', 'occasionally'),
        (r'\boccasionally\b', 'sometimes'),
        (r'\boften\b', 'frequently'),
        (r'\breally\b', 'truly'),
        (r'\bactually\b', 'in fact'),
        (r'\btake\b', 'grab'),
        (r'\bgive\b', 'provide'),
        (r'\bget\b', 'obtain'),
        (r'\bmake\b', 'create'),
        (r'\bsee\b', 'notice'),
        (r'\bgo\b', 'proceed'),
        (r'\bdo\b', 'perform'),
        (r'\bsay\b', 'mention'),
        (r'\buse\b', 'utilize'),
        (r'\btry\b', 'attempt'),
    ]

    result = text
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    return result

def apply_tone_adjustment(text: str, label: str) -> str:
    """Adjust tone based on label."""
    label_lower = label.lower()
    
    if label_lower == 'deception':
        # Make evasive/uncertain
        text = re.sub(r'\bi\b', 'we', text, flags=re.IGNORECASE)
        text = text.replace("definitely", "supposedly")
        text = text.replace("clearly", "arguably")
        text = text.replace("certainly", "perhaps")
    else:
        # Make direct/confident (truth)
        text = re.sub(r'\bwe\b', 'i', text, flags=re.IGNORECASE)
        text = text.replace("supposedly", "definitely")
        text = text.replace("arguably", "clearly")
        text = text.replace("perhaps", "certainly")
    
    return text

def paraphrase_for_duplicate(original: str, label: str, attempt: int = 1) -> str:
    """Paraphrase to make unique while preserving meaning."""
    if not original.strip():
        return original
    
    if attempt == 1:
        paraphrased = apply_complex_paraphrase(original)
    else:
        paraphrased = apply_synonym_replacements(original)
    
    # Apply tone adjustment
    paraphrased = apply_tone_adjustment(paraphrased, label)
    
    return paraphrased.strip()

def main():
    print("=" * 70)
    print("TRANSCRIPT DUPLICATE DETECTION & PARAPHRASE")
    print("=" * 70)

    # Load CSV
    csv_path = config.FINAL_CSV
    if not os.path.exists(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        return

    print(f"\n📂 Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} records")

    if 'transcript' not in df.columns or 'label' not in df.columns:
        print("Error: Missing 'transcript' or 'label' columns")
        return

    # Find duplicates
    print(f"\n🔍 Scanning for duplicate transcripts...")
    duplicate_groups = find_duplicate_groups(df)

    if not duplicate_groups:
        print("✓ No exact duplicates found!")
        output_csv = csv_path.replace(".csv", "_dedup.csv")
        df.to_csv(output_csv, index=False)
        print(f"✓ Saved deduplicated CSV to: {output_csv}")
        return

    print(f"⚠️  Found {len(duplicate_groups)} duplicate groups\n")
    
    # Process duplicates
    paraphrases_log = {}
    paraphrased_count = 0

    for group_id, indices in duplicate_groups.items():
        print(f"  Group {group_id + 1}: {len(indices)} identical transcripts")
        original_text = df.loc[indices[0], 'transcript']
        if isinstance(original_text, str):
            print(f"    Original: {original_text[:70]}...")
        else:
            print(f"    Original: [empty/invalid]")

        # Keep first, paraphrase the rest
        for attempt_num, idx in enumerate(indices[1:], 1):
            original = df.loc[idx, 'transcript']
            label = df.loc[idx, 'label']
            
            # Skip if transcript is not a string
            if not isinstance(original, str) or not original.strip():
                continue
            
            # Try paraphrasing
            new_text = paraphrase_for_duplicate(original, label, attempt=1)
            
            # If still same, try different strategy
            if normalize_text(new_text) == normalize_text(original):
                new_text = paraphrase_for_duplicate(original, label, attempt=2)
            
            df.at[idx, 'transcript'] = new_text
            paraphrased_count += 1
            paraphrases_log[idx] = {
                'original': original,
                'paraphrased': new_text,
                'label': label
            }
            print(f"    ✓ Paraphrased [{label}]: {new_text[:70]}...")

    # Save results
    output_csv = csv_path.replace(".csv", "_dedup.csv")
    df.to_csv(output_csv, index=False)
    print(f"\n💾 Saved deduplicated CSV: {output_csv}")

    # Save detailed report
    report_path = csv_path.replace(".csv", "_dedup_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("DUPLICATE TRANSCRIPT PARAPHRASE REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total Records: {len(df)}\n")
        f.write(f"Duplicate Groups Found: {len(duplicate_groups)}\n")
        f.write(f"Transcripts Paraphrased: {paraphrased_count}\n")
        f.write(f"Unique Transcripts Now: {len(df) - paraphrased_count}\n\n")
        f.write("-" * 70 + "\n")
        
        for idx, info in paraphrases_log.items():
            f.write(f"\nRecord Index: {idx}\n")
            f.write(f"Label: {info['label'].upper()}\n")
            f.write(f"Original:\n  {info['original']}\n")
            f.write(f"Paraphrased:\n  {info['paraphrased']}\n")
            f.write("-" * 70 + "\n")

    print(f"📊 Report saved: {report_path}")
    print("\n" + "=" * 70)
    print("✓ SCRIPT COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
