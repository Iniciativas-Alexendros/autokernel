import sys
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, "/home/alexendros/repositorios/org-iniciativas-alexendros/autokernel")

from autokernel.rag_index import RAGIndex


class TestRAGIndexExtra:
    """Tests adicionales para RAGIndex."""

    @pytest.fixture
    def rag(self, tmp_path):
        rag = RAGIndex()
        rag.INDEX_DIR = tmp_path
        rag.INDEX_PATH = tmp_path / "faiss.index"
        rag.METADATA_PATH = tmp_path / "metadata.json"
        rag.DOCS_DIR = tmp_path / "docs"
        yield rag

    def test_save_and_load(self, rag):
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
                fake_faiss.read_index = mock.Mock(return_value=fake_index)

                rag.add_document("text", "src")
                rag.build()
                rag.save()

                loaded = rag.load()
                assert loaded is True
