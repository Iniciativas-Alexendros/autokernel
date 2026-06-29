"""RAG index for Triton docs, kernels, specs using FAISS + bge-m3 embeddings."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    import ollama as ollama_client

    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

SCRIPT_DIR = Path(__file__).parent.parent
INDEX_DIR = SCRIPT_DIR / "workspace" / "rag"
INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.json"
DOCS_DIR = INDEX_DIR / "docs"


@dataclass
class Doc:
    text: str
    source: str
    metadata: dict


class RAGIndex:
    """FAISS-based RAG index with bge-m3 embeddings from local Ollama."""

    def __init__(self, embed_model: str = "bge-m3"):
        self.embed_model = embed_model
        self.index: Optional[faiss.IndexFlatIP] = None
        self.docs: list[Doc] = []
        self.dim = 1024  # bge-m3 output dimension

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using Ollama bge-m3."""
        if not HAS_OLLAMA:
            raise RuntimeError("ollama package not installed. Run: uv add ollama")
        vectors = []
        for text in texts:
            resp = ollama_client.embeddings(model=self.embed_model, prompt=text)
            vectors.append(resp["embedding"])
        return np.array(vectors, dtype=np.float32)

    def _chunk_text(
        self, text: str, max_chars: int = 2000, overlap: int = 200
    ) -> list[str]:
        """Split text into overlapping chunks for better retrieval."""
        if len(text) <= max_chars:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_chars
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        return chunks

    def add_document(self, text: str, source: str, metadata: dict | None = None):
        """Add a document to the index (auto-chunks large texts)."""
        chunks = self._chunk_text(text)
        for i, chunk in enumerate(chunks):
            self.docs.append(
                Doc(
                    text=chunk,
                    source=source,
                    metadata={
                        **(metadata or {}),
                        "chunk_idx": i,
                        "total_chunks": len(chunks),
                    },
                )
            )

    def build(self):
        """Build FAISS index from all added documents."""
        if not HAS_FAISS:
            raise RuntimeError("faiss-cpu not installed. Run: uv add faiss-cpu")
        if not self.docs:
            raise ValueError("No documents added. Call add_document() first.")

        texts = [d.text for d in self.docs]
        embeddings = self._embed(texts)

        self.index = faiss.IndexFlatIP(self.dim)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        self.save()

    def query(self, query_text: str, k: int = 5) -> list[Doc]:
        """Query the index for most relevant documents."""
        if self.index is None or self.index.ntotal == 0:
            self.load()

        q_embedding = self._embed([query_text])
        faiss.normalize_L2(q_embedding)

        scores, indices = self.index.search(q_embedding, min(k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = self.docs[idx]
            doc.metadata["score"] = float(score)
            results.append(doc)
        return results

    def save(self):
        """Persist index and metadata to disk."""
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(INDEX_PATH))
        metadata = [
            {"text": d.text, "source": d.source, "metadata": d.metadata}
            for d in self.docs
        ]
        with open(METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)

    def load(self) -> bool:
        """Load index from disk. Returns False if not found."""
        if not INDEX_PATH.exists() or not METADATA_PATH.exists():
            return False
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
        self.docs = [
            Doc(text=m["text"], source=m["source"], metadata=m["metadata"])
            for m in metadata
        ]
        return True


def build_default_index():
    """Build RAG index from default sources: kernels/, specs/, PROPOSAL.md."""
    rag = RAGIndex()

    # Triton kernels
    kernels_dir = SCRIPT_DIR / "kernels"
    for py_file in sorted(kernels_dir.glob("*.py")):
        if py_file.name.startswith("_") or py_file.name == "__init__.py":
            continue
        text = py_file.read_text(encoding="utf-8", errors="ignore")
        rag.add_document(
            text, f"kernels/{py_file.name}", {"type": "kernel", "lang": "triton"}
        )

    # CUDA templates
    cuda_dir = kernels_dir / "cuda"
    if cuda_dir.exists():
        for py_file in sorted(cuda_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            text = py_file.read_text(encoding="utf-8", errors="ignore")
            rag.add_document(
                text, f"kernels/cuda/{py_file.name}", {"type": "kernel", "lang": "cuda"}
            )

    # Specs
    specs_dir = SCRIPT_DIR / "specs"
    if specs_dir.exists():
        for md_file in sorted(specs_dir.rglob("*.md")):
            text = md_file.read_text(encoding="utf-8", errors="ignore")
            rel = md_file.relative_to(SCRIPT_DIR)
            rag.add_document(text, str(rel), {"type": "spec"})

    # Proposal
    proposal = SCRIPT_DIR / "PROPOSAL.md"
    if proposal.exists():
        text = proposal.read_text(encoding="utf-8", errors="ignore")
        rag.add_document(text, "PROPOSAL.md", {"type": "proposal"})

    # Existing kernels in workspace
    workspace = SCRIPT_DIR / "workspace"
    for wf in sorted(workspace.glob("kernel_*.py")):
        if "_optimized" in wf.name or "_cuda" in wf.name:
            continue
        text = wf.read_text(encoding="utf-8", errors="ignore")
        rag.add_document(text, f"workspace/{wf.name}", {"type": "optimized_kernel"})

    print(f"Building RAG index with {len(rag.docs)} chunks...")
    rag.build()
    print(f"Index saved to {INDEX_DIR}")
    return rag


if __name__ == "__main__":
    build_default_index()
