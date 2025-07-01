import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")  # Suppress transformer warnings

# 1. Load Dataset
df = pd.read_csv("translated_dataset_tagalog.csv")
print(f"Original dataset shape: {df.shape}")

# 2. Initialize Sentence Embedding Model
try:
    print("📦 Loading SentenceTransformer model (this may take a few minutes)...")
    model = SentenceTransformer("meedan/paraphrase-filipino-mpnet-base-v2")
    print("✅ Model loaded.")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit()

# 3. Clean + Prepare Columns
if not all(col in df.columns for col in ["utterance", "tagalog"]):
    raise ValueError("Missing 'utterance' or 'tagalog' columns in dataset.")

utterances = df["utterance"].fillna("").astype(str).tolist()
tagalogs = df["tagalog"].fillna("").astype(str).tolist()

# 4. Compute Similarities in Batches (with tqdm)
batch_size = 32
similarities = []

print("🔍 Scoring semantic similarity...")
for i in tqdm(range(0, len(df), batch_size), desc="Scoring"):
    batch_en = utterances[i:i+batch_size]
    batch_tl = tagalogs[i:i+batch_size]

    embeddings_en = model.encode(batch_en, convert_to_tensor=True)
    embeddings_tl = model.encode(batch_tl, convert_to_tensor=True)

    batch_sim = util.cos_sim(embeddings_en, embeddings_tl).diagonal()
    similarities.extend(batch_sim.tolist())

# 5. Append Results
df["similarity"] = similarities
threshold = 0.70
df["needs_review"] = df["similarity"] < threshold

# 6. Save to File
output_path = "evaluated_translations_with_similarity.csv"
df.to_csv(output_path, index=False)
print(f"✅ Evaluation complete! Saved to: {output_path}")
print(f"🔎 {df['needs_review'].sum()} rows flagged for review (similarity < {threshold})")
