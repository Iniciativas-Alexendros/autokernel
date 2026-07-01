import os
import sys
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autokernel.rag_index import Doc, RAGIndex


class TestRAGIndexQuery:
    """Tests unitarios para RAGIndex.query sin faiss instalado."""

    def test_build_creates_index(self):
        rag = RAGIndex()
        fake_embedding = np.ones((1, 1024), dtype=np.float32)
        fake_index = mock.Mock()
        fake_index.ntotal = 1
        fake_index.add = mock.Mock()
        with mock.patch("autokernel.rag_index.HAS_FAISS", True):
            with mock.patch("autokernel.rag_index.faiss") as fake_faiss:
                fake_faiss.IndexFlatIP = mock.Mock(return_value=fake_index)
                fake_faiss.normalize_L2 = mock.Mock()
                with mock.patch.object(rag, "_embed", return_value=fake_embedding):
                    with mock.patch.object(rag, "save"):
                        rag.add_document("hello world", "src")
                        rag.build()
        assert rag.index is fake_index
        fake_index.add.assert_called_once()

    def test_query_returns_docs_when_index_populated(self):
        rag = RAGIndex()
        rag.docs = [Doc(text="hello", source="s", metadata={})]
        fake_embedding = np.ones((1, 1024), dtype=np.float32)
        fake_index = mock.Mock()
        fake_index.ntotal = 1
        fake_index.search = mock.Mock(return_value=(np.array([[0.9]]), np.array([[0]])))
        rag.index = fake_index

        with mock.patch("autokernel.rag_index.HAS_FAISS", True):
            with mock.patch.object(rag, "_embed", return_value=fake_embedding):
                with mock.patch("autokernel.rag_index.faiss.normalize_L2"):
                    results = rag.query("hello", k=1)
        assert len(results) == 1
        assert results[0].text == "hello"
