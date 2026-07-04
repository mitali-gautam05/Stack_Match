"""
StackMatch — Embedder (Week 2)
================================
Goal: Convert every issue's prose and code into vectors (embeddings).
Then compute similarity scores using 3 approaches:

  Approach A → String similarity (no ML at all — our weakest baseline)
  Approach B → MiniLM on full text (ML but no segmentation)
  Approach C → MiniLM on prose + CodeBERT on code (our contribution)

WHY three approaches?
  We need to PROVE our dual-embedding idea is better.
  Proof requires comparison. Without baselines, we have nothing to compare to.
  This is called an "ablation study" in research.

OUTPUT: data/similarity_scores.json
  Every pair gets 3 scores → we compare which approach ranks true duplicates higher.
"""

import json
import os
import re
import time
from difflib import SequenceMatcher

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel


# ── DEVICE SETUP ───────────────────────────────────────────────────────────────
# WHY this? Models can run on CPU or GPU.
# GPU (CUDA) is 10-50x faster but not everyone has one.
# This code automatically uses GPU if available, falls back to CPU.
# "mps" = Apple Silicon GPU (M1/M2 Macs)

def get_device():
    if torch.cuda.is_available():
        print("🚀 Using GPU (CUDA)")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("🚀 Using Apple Silicon GPU (MPS)")
        return torch.device("mps")
    else:
        print("💻 Using CPU (slower but works fine for 467 pairs)")
        return torch.device("cpu")


# ── APPROACH A: STRING SIMILARITY ─────────────────────────────────────────────

def string_similarity(text_a, text_b):
    """
    Compare two texts using character-level similarity.
    No ML involved — just counting matching characters.
    
    Uses Python's built-in SequenceMatcher (Ratcliff/Obershelp algorithm).
    
    WHY include this as baseline?
    It represents "dumb" matching — if someone just ctrl+F'd for duplicates.
    Our ML models should beat this significantly.
    
    Returns: float between 0.0 and 1.0
    
    Example:
      "App crashes on startup"
      "App crash on startup"
      → 0.91  (very similar strings)
      
      "App crashes on startup"  
      "Add dark mode support"
      → 0.21  (very different strings)
    """
    if not text_a or not text_b:
        return 0.0
    
    # Normalize: lowercase, collapse whitespace
    # WHY? "App Crashes" and "app crashes" should be same
    a = re.sub(r'\s+', ' ', text_a.lower().strip())
    b = re.sub(r'\s+', ' ', text_b.lower().strip())
    
    return SequenceMatcher(None, a, b).ratio()


# ── APPROACH B: MINILM FULL TEXT EMBEDDING ────────────────────────────────────

class MiniLMEmbedder:
    """
    Embeds text using sentence-transformers MiniLM model.
    
    'all-MiniLM-L6-v2' details:
    - Trained on 1 billion sentence pairs
    - Output: 384-dimensional vector per sentence
    - Max input: 256 word pieces (truncates beyond that)
    - Size: ~80MB
    - Speed: ~14,000 sentences/second on GPU, ~2,000 on CPU
    
    WHY MiniLM and not BERT-base?
    MiniLM is a "distilled" model — smaller and faster than BERT
    but trained specifically for semantic similarity tasks.
    BERT was trained for general language understanding, not similarity.
    For our use case, MiniLM > BERT.
    
    WHY not GPT embeddings?
    - Cost: OpenAI charges per token
    - Speed: API calls are slow
    - Offline: We want to run without internet after model download
    MiniLM is free, fast, and runs locally.
    """
    
    def __init__(self):
        print("\nLoading MiniLM model...")
        # This downloads ~80MB on first run, then caches locally
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ MiniLM loaded")
    
    def embed(self, text):
        """
        Convert text to a 384-dimensional vector.
        
        encode() does:
        1. Tokenize text into word pieces
        2. Pass through 6 transformer layers
        3. Pool all token outputs into one vector (mean pooling)
        4. Normalize to unit length (so cosine sim = dot product)
        
        normalize_embeddings=True → vector length = 1.0
        WHY normalize? When vectors are unit length, 
        cosine_similarity(a,b) = dot_product(a,b)
        Dot product is faster to compute than full cosine formula.
        """
        if not text or not text.strip():
            return np.zeros(384)  # Zero vector for empty text
        
        return self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False
        )
    
    def similarity(self, text_a, text_b):
        """Embed both texts and compute cosine similarity."""
        vec_a = self.embed(text_a)
        vec_b = self.embed(text_b)
        
        # Cosine similarity = dot product of unit vectors
        # Result is between -1 and 1, but for text usually 0 to 1
        return float(np.dot(vec_a, vec_b))
    
    def embed_batch(self, texts, batch_size=32):
        """
        Embed multiple texts at once — much faster than one by one.
        
        WHY batching?
        GPU processes multiple inputs in parallel.
        Embedding 32 texts at once takes almost the same time as embedding 1.
        For 467 pairs (934 issues), batching gives ~30x speedup.
        """
        # Filter empty texts, keep track of indices
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
        
        if not valid_texts:
            return np.zeros((len(texts), 384))
        
        # Encode all valid texts in batches
        embeddings = self.model.encode(
            valid_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        
        # Put results back in correct positions
        result = np.zeros((len(texts), 384))
        for i, idx in enumerate(valid_indices):
            result[idx] = embeddings[i]
        
        return result


# ── APPROACH C: CODEBERT ──────────────────────────────────────────────────────

class CodeBERTEmbedder:
    """
    Embeds code using Microsoft's CodeBERT model.
    
    CodeBERT details:
    - Trained on 6 programming languages: Python, Java, JavaScript, PHP, Ruby, Go
    - Also trained on natural language + code pairs (bimodal pretraining)
    - Output: 768-dimensional vector
    - Max input: 512 tokens
    - Size: ~500MB
    
    WHY CodeBERT specifically?
    CodeBERT was trained with TWO objectives:
    1. Masked Language Modeling (like BERT) → understands code structure
    2. Replaced Token Detection → understands which tokens "belong" together
    
    This means it understands:
    - Variable names and their relationships
    - Function signatures
    - Error messages and stack traces
    - The MEANING of code, not just its text
    
    MiniLM on "AttributeError: NoneType" → treats it like English words
    CodeBERT on "AttributeError: NoneType" → knows this is a Python exception
    """
    
    def __init__(self, device):
        print("\nLoading CodeBERT model (~500MB, may take a minute)...")
        self.device = device
        
        model_name = "microsoft/codebert-base"
        
        # Tokenizer: converts code text → token IDs
        # AutoTokenizer automatically downloads the right tokenizer for the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Model: converts token IDs → embeddings
        # AutoModel loads the base model (no classification head — we want embeddings)
        self.model = AutoModel.from_pretrained(model_name)
        self.model = self.model.to(device)  # Move to GPU if available
        self.model.eval()  # Inference mode — disables dropout, faster
        
        print("✓ CodeBERT loaded")
    
    def embed(self, code_text):
        """
        Convert code text to a 768-dimensional vector.
        
        HOW it works internally:
        1. Tokenize: "def load(id):" → ["def", "load", "(", "id", ")", ":"]
        2. Add special tokens: [CLS] def load ( id ) : [SEP]
           [CLS] = "start of sequence" token
           [SEP] = "end of sequence" token
        3. Truncate to max 512 tokens (CodeBERT's limit)
        4. Pass through 12 transformer layers
        5. Take the [CLS] token's output as the sequence embedding
           WHY [CLS]? It's designed to capture overall sequence meaning
        6. Normalize to unit length
        
        WHY mean pooling not [CLS]?
        Actually both work. For code, [CLS] pooling tends to capture
        overall "what does this code do" meaning better.
        For prose, mean pooling (what MiniLM does) tends to work better.
        Different tasks, different pooling strategies.
        """
        if not code_text or not code_text.strip():
            return np.zeros(768)
        
        # Tokenize with truncation at 512 tokens
        # return_tensors="pt" → return PyTorch tensors (not numpy)
        inputs = self.tokenizer(
            code_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        # Move inputs to same device as model (GPU/CPU)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # torch.no_grad() → don't compute gradients
        # WHY? Gradients are needed for training (backprop).
        # During inference we don't train, so we skip gradient computation.
        # This saves ~50% memory and speeds up inference.
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # outputs.last_hidden_state shape: [batch_size, seq_len, 768]
        # [:, 0, :] → take [CLS] token (index 0) for all batches
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        
        # Move back to CPU and convert to numpy
        embedding = cls_embedding.cpu().numpy()
        
        # Normalize to unit vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def similarity(self, code_a, code_b):
        """Embed both code snippets and compute cosine similarity."""
        if not code_a.strip() and not code_b.strip():
            return 0.0  # Both empty → no code signal
        
        if not code_a.strip() or not code_b.strip():
            return 0.0  # One empty → can't compare
        
        vec_a = self.embed(code_a)
        vec_b = self.embed(code_b)
        
        return float(np.dot(vec_a, vec_b))


# ── FUSION ─────────────────────────────────────────────────────────────────────

def fuse_scores(prose_sim, code_sim, has_code_a, has_code_b, alpha=0.5):
    """
    Combine prose similarity and code similarity into one final score.
    
    alpha = weight for prose (1-alpha = weight for code)
    alpha=0.5 means equal weight to both.
    
    WHY weighted average and not just concatenate vectors?
    Option 1: Concatenate vectors → [prose_vec | code_vec] → 1536 dims
              Then compute cosine similarity of concatenated vectors.
              Problem: if code is empty (zeros), it drags down similarity.
    
    Option 2: Weighted average of SCORES (what we do here)
              More interpretable, easier to tune alpha.
              If no code → fall back to prose only.
    
    This is a design choice — both are valid. We chose Option 2 for simplicity
    and explainability. A reviewer can understand "70% prose + 30% code".
    
    WHY alpha=0.5 as default?
    Starting point. We can tune this later using our evaluation metrics.
    Maybe prose matters more → alpha=0.7. We'll find out from data.
    """
    both_have_code = has_code_a and has_code_b
    
    if not both_have_code:
        # If either issue has no code → use prose similarity only
        # WHY? Comparing a code vector to a zero vector is meaningless.
        return prose_sim
    
    # Both have code → weighted combination
    return alpha * prose_sim + (1 - alpha) * code_sim


# ── MAIN PIPELINE ──────────────────────────────────────────────────────────────

def compute_all_scores(data_path="data/segmented_pairs.json",
                       output_path="data/similarity_scores.json"):
    """
    For every pair in our dataset, compute similarity using all 3 approaches.
    Save scores to JSON for evaluation step.
    """
    
    print("=" * 60)
    print("StackMatch — Computing Similarity Scores")
    print("=" * 60)
    
    # Load data
    with open(data_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    print(f"\nLoaded {len(pairs)} pairs")
    
    # Setup
    device = get_device()
    minilm = MiniLMEmbedder()
    codebert = CodeBERTEmbedder(device)
    
    results = []
    
    print(f"\nProcessing {len(pairs)} pairs...")
    print("(This will take 10-20 minutes on CPU, ~2 min on GPU)\n")
    
    start_time = time.time()
    
    for i, pair in enumerate(pairs):
        dup = pair["duplicate"]
        orig = pair["original"]
        
        # ── Approach A: String Similarity ──────────────────────────────
        # Use full raw body text (title + body combined)
        dup_full = dup["title"] + " " + dup["body"]
        orig_full = orig["title"] + " " + orig["body"]
        score_a = string_similarity(dup_full, orig_full)
        
        # ── Approach B: MiniLM on Full Text ────────────────────────────
        # No segmentation — treat entire issue as one text
        score_b = minilm.similarity(dup_full, orig_full)
        
        # ── Approach C: Dual Embedding ─────────────────────────────────
        # Prose → MiniLM, Code → CodeBERT, then fuse
        prose_sim = minilm.similarity(dup["prose"], orig["prose"])
        
        has_code_dup = bool(dup["code"].strip())
        has_code_orig = bool(orig["code"].strip())
        
        code_sim = codebert.similarity(dup["code"], orig["code"])
        
        score_c = fuse_scores(
            prose_sim, code_sim,
            has_code_dup, has_code_orig,
            alpha=0.5
        )
        
        # ── Save Result ────────────────────────────────────────────────
        results.append({
            "repo": pair["repo"],
            "duplicate_number": dup["number"],
            "original_number": orig["number"],
            "issue_type": dup["issue_type"],
            "has_code": has_code_dup and has_code_orig,
            
            # The three scores — this is what we'll evaluate
            "score_string":    round(score_a, 4),
            "score_minilm":    round(score_b, 4),
            "score_dual":      round(score_c, 4),
            
            # Sub-scores for analysis
            "prose_similarity": round(prose_sim, 4),
            "code_similarity":  round(code_sim, 4),
        })
        
        # Progress update
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(pairs) - i - 1) / rate
            print(f"  [{i+1}/{len(pairs)}] "
                  f"~{remaining:.0f}s remaining | "
                  f"Last scores: "
                  f"string={score_a:.3f} "
                  f"minilm={score_b:.3f} "
                  f"dual={score_c:.3f}")
    
    # Save
    os.makedirs("data", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Saved {len(results)} scored pairs → {output_path}")
    print_score_stats(results)
    
    return results


def print_score_stats(results):
    """Quick summary of what scores look like."""
    import statistics
    
    string_scores = [r["score_string"] for r in results]
    minilm_scores = [r["score_minilm"] for r in results]
    dual_scores   = [r["score_dual"]   for r in results]
    
    print("\n── Score Distribution (higher = more similar) ─────")
    print(f"{'Approach':<20} {'Mean':>8} {'Median':>8} {'Min':>8} {'Max':>8}")
    print("-" * 56)
    
    for name, scores in [
        ("String Similarity", string_scores),
        ("MiniLM (full text)", minilm_scores),
        ("Dual Embedding",    dual_scores),
    ]:
        print(f"{name:<20} "
              f"{statistics.mean(scores):>8.3f} "
              f"{statistics.median(scores):>8.3f} "
              f"{min(scores):>8.3f} "
              f"{max(scores):>8.3f}")
    
    # Code-heavy pairs separately
    code_pairs = [r for r in results if r["has_code"]]
    if code_pairs:
        print(f"\nCode-present pairs ({len(code_pairs)} pairs):")
        print(f"  Dual vs MiniLM improvement: "
              f"{(sum(r['score_dual'] for r in code_pairs) - sum(r['score_minilm'] for r in code_pairs)) / len(code_pairs):+.4f} avg")
    
    print("───────────────────────────────────────────────────")


if __name__ == "__main__":
    compute_all_scores()