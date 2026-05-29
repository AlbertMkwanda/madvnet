import os
import random
import re
from pathlib import Path

import pandas as pd


def parse_duration_to_seconds(value):
    if pd.isna(value):
        return 0.0
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0

    # Numeric values are already seconds
    if re.match(r'^[0-9]+(?:\.[0-9]+)?$', text):
        return float(text)

    parts = text.split(':')
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    raise ValueError(f"Cannot parse duration value: {value}")


def normalize_time_columns(df):
    for col in ['start_time', 'end_time']:
        if col in df.columns:
            df[col] = df[col].apply(parse_duration_to_seconds).astype(float)
    return df


def build_synonym_replacements(seed: int):
    rnd = random.Random(seed)
    replacements = [
        (r"\bdo not\b", ["don't", "do n't"]),
        (r"\bdid not\b", ["didn't", "did not"]),
        (r"\bcan not\b", ["can't", "cannot"]),
        (r"\bwill not\b", ["won't", "will not"]),
        (r"\bgoing to\b", ["gonna", "going to"]),
        (r"\bi am\b", ["i'm", "i am"]),
        (r"\bthank you\b", ["thanks", "thank you"]),
        (r"\bbecause\b", ["since", "because"]),
        (r"\bbefore\b", ["prior to", "before"]),
        (r"\bthen\b", ["after that", "then"]),
        (r"\bsaid\b", ["stated", "said"]),
        (r"\basked\b", ["asked", "inquired"]),
        (r"\bi think\b", ["i believe", "i think"]),
        (r"\blooks like\b", ["seems like", "looks like"]),
        (r"\bit was\b", ["it was", "it felt like"]),
        (r"\bit is\b", ["it's", "it is"]),
        (r"\bdo you\b", ["are you", "do you"]),
    ]
    return [(pattern, rnd.choice(choices)) for pattern, choices in replacements]


def paraphrase_transcript(text: str, seed: int = 0) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    rnd = random.Random(seed)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    # Keep transcripts lowercase and clean whitespace.
    text = text.lower()

    # Replace several patterns with selectable paraphrases.
    for pattern, replacement in build_synonym_replacements(seed):
        if re.search(pattern, text):
            text = re.sub(pattern, replacement, text)

    # Replace some longer phrases if they exist.
    phrases = [
        (r"\band then\b", "then"),
        (r"\band i\b", "i also"),
        (r"\band he\b", "he also"),
        (r"\band she\b", "she also"),
        (r"\band they\b", "they also"),
        (r"\balright\b", "okay"),
        (r"\bjust\b", "simply"),
    ]
    for pattern, replacement in phrases:
        if rnd.random() < 0.35:
            text = re.sub(pattern, replacement, text)

    # Apply small random sentence-level shifts.
    if rnd.random() < 0.3:
        text = text.replace("? ", ". ")
    if rnd.random() < 0.2:
        text = text.replace("! ", ". ")

    text = re.sub(r"\s+([\.,\?\!])", r"\1", text)
    text = text.strip()
    return text


def oversample_rldd(final_csv_path: str, output_path: str = None, random_seed: int = 42):
    df = pd.read_csv(final_csv_path)
    df = normalize_time_columns(df)

    rldd_mask = df['YT_Video_ID'].astype(str).str.contains('RLDD', case=False, na=False)
    rldd_df = df[rldd_mask].copy()
    other_df = df[~rldd_mask].copy()

    if rldd_df.empty:
        raise ValueError('No RLDD rows found in the dataset.')

    target_rldd_count = len(other_df)
    current_rldd_count = len(rldd_df)
    if target_rldd_count <= current_rldd_count:
        raise ValueError(
            f'RLDD already has {current_rldd_count} rows and target <= current count ({target_rldd_count}).'
        )

    oversampled_rows = [row.to_dict() for _, row in rldd_df.iterrows()]
    rnd = random.Random(random_seed)
    while len(oversampled_rows) < target_rldd_count:
        original = rldd_df.sample(n=1, random_state=rnd.randint(0, 2**31 - 1)).iloc[0]
        copy_row = original.copy()
        copy_row['transcript'] = paraphrase_transcript(original['transcript'], seed=rnd.randint(0, 2**31 - 1))
        oversampled_rows.append(copy_row.to_dict())

    oversampled_df = pd.DataFrame(oversampled_rows)
    combined_df = pd.concat([other_df.reset_index(drop=True), oversampled_df.reset_index(drop=True)], ignore_index=True)
    output_path = output_path or os.path.splitext(final_csv_path)[0] + '_oversampled.csv'
    combined_df.to_csv(output_path, index=False)

    print(f'Oversampled RLDD from {current_rldd_count} to {len(oversampled_df)} rows.')
    print(f'Combined dataset rows: {len(combined_df)}')
    print(f'Saved combined DOLOS + oversampled RLDD CSV at: {output_path}')
    return output_path


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent.parent
    input_csv = base_dir / 'data' / 'dolos_final_training_retranscribed_medium.csv'
    output_csv = base_dir / 'data' / 'dolos_final_training_retranscribed_medium_oversampled.csv'
    oversample_rldd(str(input_csv), str(output_csv), random_seed=2026)
