import os
import sys
from unittest import mock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autokernel.llm_assistant import LLMAssistant, Spec


class TestLLMAssistantExtra:
    """Tests adicionales para LLMAssistant."""

    @pytest.fixture
    def assistant(self):
        with mock.patch("autokernel.llm_assistant.ollama_client") as fake_ollama:
            fake_ollama.chat = mock.Mock(return_value={"message": {"content": "spec json"}})
            fake_ollama.embeddings = mock.Mock(return_value={"embedding": [0.1] * 1024})
            fake_ollama.stop = mock.Mock()
            with mock.patch("autokernel.llm_assistant.RAGIndex") as fake_rag:
                fake_rag.return_value.query = mock.Mock(return_value=[])
                yield LLMAssistant()

    def test_generate_spec_returns_spec(self, assistant):
        with mock.patch(
            "autokernel.llm_assistant.LLMAssistant._call_ollama",
            return_value='{"kernel_type": "matmul", "optimizations": ["tiling"]}',
        ):
            spec = assistant.generate_spec("matmul", {"time_ms": 10.0})
            assert isinstance(spec, Spec)
            assert spec.kernel_type == "matmul"

    def test_generate_kernel(self, assistant):
        spec = Spec(kernel_type="matmul", strategies=[{"name": "tiling"}])
        with mock.patch(
            "autokernel.llm_assistant.LLMAssistant._call_ollama",
            return_value="```python\nKERNEL_TYPE = 'matmul'\n```",
        ):
            code = assistant.generate_kernel(spec, "tests")
            assert "KERNEL_TYPE = 'matmul'" in code

    def test_generate_tests(self, assistant):
        spec = Spec(kernel_type="matmul", strategies=[{"name": "tiling"}])
        with mock.patch(
            "autokernel.llm_assistant.LLMAssistant._call_ollama",
            return_value="```python\ndef test_matmul():\n    pass\n```",
        ):
            code = assistant.generate_tests(spec)
            assert "def test_matmul" in code
