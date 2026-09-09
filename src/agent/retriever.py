"""
Historical Resolution Retriever for Amazon Customer Support.
Performs dense semantic search over historical AmazonHelp customer-reply pairs
to retrieve the most relevant verified resolutions.
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any
import numpy as np


KB_FILE = Path("data/amazon_knowledge_base.jsonl")
EMBEDDINGS_CACHE = Path("data/kb_embeddings.npy")


class HistoricalRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", max_docs: int = 5000):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self._load_knowledge_base(max_docs)
        self._load_or_compute_embeddings()

    def _load_knowledge_base(self, max_docs: int):
        if not KB_FILE.exists():
            raise FileNotFoundError(f"Knowledge base file {KB_FILE} not found. Run build_knowledge_base.py first.")
        with open(KB_FILE, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= max_docs:
                    break
                self.documents.append(json.loads(line))

    def _load_or_compute_embeddings(self):
        if EMBEDDINGS_CACHE.exists():
            self.embeddings = np.load(EMBEDDINGS_CACHE)
            if len(self.embeddings) == len(self.documents):
                return

        texts = [doc["customer_text_clean"] for doc in self.documents]
        self.embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=64
        )
        EMBEDDINGS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.save(EMBEDDINGS_CACHE, self.embeddings)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        # Clean query
        query_clean = re.sub(r"@[A-Za-z0-9_]+", "", query)
        query_clean = re.sub(r"https?://\S+", "", query_clean)
        query_clean = re.sub(r"\s+", " ", query_clean).strip()

        if not query_clean:
            return []

        query_emb = self.model.encode([query_clean], normalize_embeddings=True, show_progress_bar=False)[0]
        # Cosine similarity
        sims = np.dot(self.embeddings, query_emb)
        top_indices = np.argsort(-sims)[:top_k]

        results = []
        for idx in top_indices:
            doc = dict(self.documents[idx])
            doc["similarity_score"] = float(sims[idx])
            results.append(doc)
        return results
