from transformers import pipeline
import pandas as pd
from tqdm import tqdm  # Progress bar
import warnings
warnings.filterwarnings("ignore")  # Suppress tokenizer warnings

# 1. Load Dataset
df = pd.read_csv("dataset/Bitext_Sample_Customer_Service_Training_Dataset.csv")
print(f"Original dataset shape: {df.shape}")

# 2. Initialize Translator (with error handling)
try:
    translator = pipeline(
        "translation",
        model="Helsinki-NLP/opus-mt-en-tl",
        device="cpu"  # Use "cuda" if you have GPU
    )
except Exception as e:
    print(f"Failed to load model: {e}")
    exit()

# 3. Optimized Translation Function
def translate_to_tagalog(text):
    try:
        if pd.isna(text) or str(text).strip() == "":
            return ""
        result = translator(text, max_length=100, truncation=True)[0]["translation_text"]
        return result.replace("Ã±", "ñ").replace("Ã¯", "ï")  # Fix common encoding errors
    except Exception as e:
        print(f"Error translating '{text}': {e}")
        return "TRANSLATION_ERROR"

# 4. Batch Translation (with progress bar)
batch_size = 40  # Reduce if you get timeout errors
tagalog_translations = []
for i in tqdm(range(0, len(df), batch_size)):
    batch = df["utterance"].iloc[i:i+batch_size].tolist()
    translated = [translate_to_tagalog(text) for text in batch]
    tagalog_translations.extend(translated)

df["tagalog"] = tagalog_translations

# 5. Save Results
df.to_csv("translated_dataset_tagalog.csv", index=False)
print("Translation complete! Saved to 'translated_dataset_tagalog.csv'")