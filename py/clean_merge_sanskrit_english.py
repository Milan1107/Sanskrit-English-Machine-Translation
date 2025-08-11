import pandas as pd
import unicodedata
import re
from pathlib import Path

def normalize_text(text):
    """Normalize Unicode and strip spaces"""
    return unicodedata.normalize("NFC", text.strip())

def is_sanskrit(text):
    """Check if text contains Devanagari characters"""
    return any("\u0900" <= ch <= "\u097F" for ch in text)

def is_english(text):
    """Check if text contains Latin letters"""
    return bool(re.search(r"[A-Za-z]", text))

# Folder containing .sa and .en files
data_folder = Path("../datasets/")
output_file = "cleaned_dataset.csv"

datasets = []

# Match lowercase and uppercase extensions
for sa_file in list(data_folder.glob("*.sa")) + list(data_folder.glob("*.SA")):
    en_file = sa_file.with_suffix(".en")
    if not en_file.exists():
        en_file = sa_file.with_suffix(".EN")
    if not en_file.exists():
        print(f"⚠ No matching English file for {sa_file.name}")
        continue

    with open(sa_file, encoding="utf-8") as f:
        sans_lines = [normalize_text(l) for l in f if l.strip()]
    with open(en_file, encoding="utf-8") as f:
        eng_lines = [normalize_text(l) for l in f if l.strip()]

    min_len = min(len(sans_lines), len(eng_lines))
    sans_lines = sans_lines[:min_len]
    eng_lines = eng_lines[:min_len]

    cleaned_pairs = []
    for s, e in zip(sans_lines, eng_lines):
        # Fix swapped cases
        if is_english(s) and is_sanskrit(e):
            s, e = e, s
        
        # Skip lines if they are wrong language after fix
        if not (is_sanskrit(s) and is_english(e)):
            continue

        # Skip very short lines
        if len(s.split()) <= 1 or len(e.split()) <= 1:
            continue

        cleaned_pairs.append((s, e))

    if cleaned_pairs:
        df = pd.DataFrame(cleaned_pairs, columns=["sanskrit", "english"])
        datasets.append(df)
        print(f"✅ Processed {sa_file.name} ({len(df)} pairs)")

if datasets:
    final_df = pd.concat(datasets, ignore_index=True)
    final_df.drop_duplicates(inplace=True)
    final_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\n🎯 Final cleaned dataset saved to {output_file} with {len(final_df)} pairs.")
else:
    print("❌ No valid dataset found.")
