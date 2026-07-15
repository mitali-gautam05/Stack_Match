"""
StackMatch — Embedder v2
=========================
Change from v1: Save actual embedding VECTORS, not just similarity scores.

WHY save vectors?
  v1 saved: score_minilm = 0.821  (one number per pair)
  v2 saves: prose_vec = [0.12, -0.34, 0.87, ...]  (384 numbers per issue)

With actual vectors we can do REAL retrieval:
  dup_42 vs ALL 467 originals → proper ranking → meaningful MRR

Output files:
  data/embeddings.npz  → numpy arrays (compact binary format)
  data/metadata.json   → issue numbers, repos, types (for lookup)
"""

import json
import os
import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel


# ── DEVICE ─────────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        print("🚀 GPU (CUDA)")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("🚀 Apple Silicon (MPS)")
        return torch.device("mps")
    else:
        print("💻 CPU")
        return torch.device("cpu")


# ── MINILM ─────────────────────────────────────────────────────────────────────
class MiniLMEmbedder:
    def __init__(self):
        print("Loading MiniLM...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ MiniLM ready (384-dim)")

    def embed_batch(self, texts, batch_size=64):
        """
        Embed list of texts → numpy array of shape (N, 384)
        Empty texts get zero vectors.
        """
        results = np.zeros((len(texts), 384), dtype=np.float32)
        
        valid = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        if not valid:
            return results
        
        indices, valid_texts = zip(*valid)
        
        vecs = self.model.encode(
            list(valid_texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        
        for i, idx in enumerate(indices):
            results[idx] = vecs[i]
        
        return results


# ── CODEBERT ───────────────────────────────────────────────────────────────────
class CodeBERTEmbedder:
    def __init__(self, device):
        print("Loading CodeBERT (~500MB)...")
        self.device = device
        model_name = "microsoft/codebert-base"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        print("✓ CodeBERT ready (768-dim)")

    def embed_single(self, text):
        if not text or not text.strip():
            return np.zeros(768, dtype=np.float32)
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        vec = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        norm = np.linalg.norm(vec)
        return (vec / norm).astype(np.float32) if norm > 0 else vec

    def embed_batch(self, texts, batch_size=16):
        """
        Embed list of code texts → numpy array of shape (N, 768)
        Smaller batch_size than MiniLM because CodeBERT is larger.
        """
        results = np.zeros((len(texts), 768), dtype=np.float32)
        
        print(f"  Embedding {len(texts)} code snippets...")
        for i, text in enumerate(texts):
            results[i] = self.embed_single(text)
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(texts)}]")
        
        return results


# ── MAIN ───────────────────────────────────────────────────────────────────────
def embed_all(data_path="data/segmented_pairs.json",
              output_dir="data"):
    """
    Embed all issues and save vectors to disk.
    
    WHY .npz format?
      numpy's compressed format — much faster to load than JSON
      467 × 384 floats as JSON = ~3MB text, slow to parse
      467 × 384 floats as .npz = ~700KB binary, instant load
    
    Structure saved:
      dup_prose_vecs  : (467, 384) — duplicate issue prose embeddings
      dup_code_vecs   : (467, 768) — duplicate issue code embeddings
      orig_prose_vecs : (467, 384) — original issue prose embeddings
      orig_code_vecs  : (467, 768) — original issue code embeddings
    """
    
    print("=" * 60)
    print("StackMatch — Embedding Pipeline v2")
    print("=" * 60)
    
    with open(data_path, encoding="utf-8") as f:
        pairs = json.load(f)
    print(f"\nLoaded {len(pairs)} pairs")
    
    # Extract texts
    dup_prose  = [p["duplicate"]["prose"] for p in pairs]
    dup_code   = [p["duplicate"]["code"]  for p in pairs]
    orig_prose = [p["original"]["prose"]  for p in pairs]
    orig_code  = [p["original"]["code"]   for p in pairs]
    
    # Metadata for lookup later
    metadata = [{
        "repo": p["repo"],
        "dup_number":  p["duplicate"]["number"],
        "orig_number": p["original"]["number"],
        "issue_type":  p["duplicate"]["issue_type"],
        "has_code": bool(p["duplicate"]["code"].strip() and
                        p["original"]["code"].strip())
    } for p in pairs]
    
    device = get_device()
    
    # ── Embed Prose with MiniLM ──────────────────────────────────────
    minilm = MiniLMEmbedder()
    
    print("\n[1/4] Embedding duplicate prose...")
    dup_prose_vecs = minilm.embed_batch(dup_prose)
    
    print("\n[2/4] Embedding original prose...")
    orig_prose_vecs = minilm.embed_batch(orig_prose)
    
    # ── Embed Code with CodeBERT ─────────────────────────────────────
    codebert = CodeBERTEmbedder(device)
    
    print("\n[3/4] Embedding duplicate code...")
    dup_code_vecs = codebert.embed_batch(dup_code)
    
    print("\n[4/4] Embedding original code...")
    orig_code_vecs = codebert.embed_batch(orig_code)
    
    # ── Save ─────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    
    npz_path = os.path.join(output_dir, "embeddings.npz")
    np.savez_compressed(
        npz_path,
        dup_prose_vecs  = dup_prose_vecs,
        dup_code_vecs   = dup_code_vecs,
        orig_prose_vecs = orig_prose_vecs,
        orig_code_vecs  = orig_code_vecs
    )
    
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Embeddings saved → {npz_path}")
    print(f"✅ Metadata saved  → {meta_path}")
    print(f"\nShapes:")
    print(f"  dup_prose_vecs:  {dup_prose_vecs.shape}")
    print(f"  dup_code_vecs:   {dup_code_vecs.shape}")
    print(f"  orig_prose_vecs: {orig_prose_vecs.shape}")
    print(f"  orig_code_vecs:  {orig_code_vecs.shape}")
    
    # Quick sanity check
    print(f"\nSanity check — first pair:")
    sim = float(np.dot(dup_prose_vecs[0], orig_prose_vecs[0]))
    print(f"  Prose similarity (pair 0): {sim:.4f}  (should be > 0.3)")


if __name__ == "__main__":
    embed_all()