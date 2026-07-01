import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autokernel.rag_index import RAGIndex


class TestRAGIndex:
    """Tests para RAGIndex con mocks de FAISS y Ollama."""

    @pytest.fixture
    def rag(self, tmp_path):
        with tempfile.TemporaryDirectory() as td:
            index_dir = Path(td)
            rag = RAGIndex()
            rag.INDEX_DIR = index_dir
            rag.INDEX_PATH = index_dir / "faiss.index"
            rag.METADATA_PATH = index_dir / "metadata.json"
            rag.DOCS_DIR = index_dir / "docs"
            yield rag

    def test_add_document_chunks(self, rag):
        rag.add_document("texto corto", "src")
        assert len(rag.docs) == 1

    def test_chunk_text_splits_long_text(self, rag):
        text = "a" * 2500
        chunks = rag._chunk_text(text, max_chars=1000, overlap=100)
        assert len(chunks) > 1
        assert all(len(c) <= 1000 for c in chunks)

    def test_build_and_query(self, rag):
        fake_embedding = np.ones((1, 1024), dtype=np.float32)

        with mock.patch("autokernel.rag_index.ollama_client") as fake_ollama:
            fake_ollama.embeddings = mock.Mock(
                return_value={"embedding": fake_embedding[0].tolist()}
            )
            with (
                mock.patch("autokernel.rag_index.HAS_FAISS", True),
                mock.patch("autokernel.rag_index.faiss") as fake_faiss,
            ):
                fake_index = mock.Mock()
                fake_index.search = mock.Mock(return_value=(np.array([[1.0]]), np.array([[0]])))
                fake_index.ntotal = 1
                fake_faiss.IndexFlatIP = mock.Mock(return_value=fake_index)
                fake_faiss.normalize_L2 = mock.Mock()
                fake_faiss.write_index = mock.Mock()

                rag.add_document("kernel matmul", "kernels/matmul.py")
                rag.build()
                fake_ollama.embeddings.assert_called_once()

                results = rag.query("matmul optimization", k=1)
                assert len(results) == 1
