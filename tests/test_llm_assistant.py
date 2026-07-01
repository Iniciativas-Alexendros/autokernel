import os
import sys
from unittest import mock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autokernel.llm_assistant import LLMAssistant


class TestLLMAssistant:
    """Tests para LLMAssistant."""

    @pytest.fixture
    def assistant(self):
        with mock.patch("autokernel.llm_assistant.ollama_client") as fake_ollama:
            fake_ollama.chat = mock.Mock(return_value={"message": {"content": "plan"}})
            fake_ollama.embeddings = mock.Mock(return_value={"embedding": [0.1] * 1024})
            fake_ollama.stop = mock.Mock()
            with mock.patch("autokernel.llm_assistant.RAGIndex") as fake_rag:
                fake_rag.return_value.query = mock.Mock(return_value=[])
                yield LLMAssistant()

    def test_call_ollama(self, assistant):
        result = assistant._call_ollama("ornith:9b", [{"role": "user", "content": "hola"}])
        assert result == "plan"

    def test_switch_model(self, assistant):
        assistant._switch_model("model-a")
        assert assistant._current_model == "model-a"

    def test_opencode_prefix_routes_to_api(self, assistant):
        with (
            mock.patch(
                "autokernel.llm_assistant.LLMAssistant._read_openrouter_key",
                return_value="fake-key",
            ),
            mock.patch("urllib.request.urlopen") as fake_urlopen,
        ):
            fake_resp = mock.Mock()
            fake_resp.read = mock.Mock(
                return_value=b'{"choices": [{"message": {"content": "ok"}}]}'
            )
            fake_urlopen.return_value.__enter__ = mock.Mock(return_value=fake_resp)
            fake_urlopen.return_value.__exit__ = mock.Mock(return_value=False)
            result = assistant._call_ollama("opencode/mimo-v2.5-free", [])
            assert result == "ok"
