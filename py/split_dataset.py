import pandas as pd
from sklearn.model_selection import train_test_split

# Load cleaned dataset
df = pd.read_csv("cleaned_dataset.csv")

print(f"Total sentence pairs: {len(df)}")

# Train/Validation/Test split (80/10/10)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

# Save CSV versions (optional but useful for debugging/analysis)
train_df.to_csv("train.csv", index=False, encoding="utf-8")
val_df.to_csv("val.csv", index=False, encoding="utf-8")
test_df.to_csv("test.csv", index=False, encoding="utf-8")

# Save parallel text files
def save_parallel(df, prefix):
    with open(f"{prefix}.sa", "w", encoding="utf-8") as f_sa, \
         open(f"{prefix}.en", "w", encoding="utf-8") as f_en:
        for _, row in df.iterrows():
            f_sa.write(row["sanskrit"] + "\n")
            f_en.write(row["english"] + "\n")

save_parallel(train_df, "train")
save_parallel(val_df, "val")
save_parallel(test_df, "test")

print("✅ Dataset split and saved as .sa/.en files (and CSVs).")
