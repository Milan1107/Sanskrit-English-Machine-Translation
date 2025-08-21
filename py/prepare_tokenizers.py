import sentencepiece as spm
from pathlib import Path

# ------------------ Paths ------------------
data_folder = Path("./sa-en/")
train_sanskrit_file = data_folder/"train.sa"
train_english_file = data_folder/"train.en"
val_sanskrit_file = data_folder/"val.sa"
val_english_file = data_folder/"val.en"
test_sanskrit_file = data_folder/"test.sa"
test_english_file = data_folder/"test.en"

# ------------------ Step 1: Train SentencePiece Tokenizers ------------------
print("Training Sanskrit tokenizer...")
spm.SentencePieceTrainer.Train(
    input=str(train_sanskrit_file),
    model_prefix=str(data_folder / "sp_sanskrit"),
    vocab_size=16000,
    model_type='bpe'
)

print("Training English tokenizer...")
spm.SentencePieceTrainer.Train(
    input=str(train_english_file),
    model_prefix=str(data_folder / "sp_english"),
    vocab_size=16000,
    model_type='bpe'
)

# ------------------ Step 2: Load trained tokenizers ------------------
sp_sanskrit = spm.SentencePieceProcessor(model_file=str(data_folder / "sp_sanskrit.model"))
sp_english = spm.SentencePieceProcessor(model_file=str(data_folder / "sp_english.model"))

# ------------------ Step 3: Encode sentences into token IDs ------------------
def encode_file(input_file, tokenizer, output_file):
    with open(input_file, encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            if line:
                token_ids = tokenizer.encode(line)  # list of integers
                f_out.write(" ".join(map(str, token_ids)) + "\n")

# Encode all splits
print("Encoding training data...")
encode_file(train_sanskrit_file, sp_sanskrit, data_folder / "train_sanskrit_ids.txt")
encode_file(train_english_file, sp_english, data_folder / "train_english_ids.txt")

print("Encoding validation data...")
encode_file(val_sanskrit_file, sp_sanskrit, data_folder / "val_sanskrit_ids.txt")
encode_file(val_english_file, sp_english, data_folder / "val_english_ids.txt")

print("Encoding test data...")
encode_file(test_sanskrit_file, sp_sanskrit, data_folder / "test_sanskrit_ids.txt")
encode_file(test_english_file, sp_english, data_folder / "test_english_ids.txt")

print("✅ Tokenization and encoding completed. Files ready for embedding layers.")
