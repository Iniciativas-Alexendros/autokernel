import os
import sys

import pytest

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO_ROOT)


class TestLLMRouter:
    """Tests del router de modelos LLM."""

    def test_llm_assistant_accepts_opencode_models(self):
        """El asistente puede instanciarse con modelos opencode como fallback."""
        from autokernel.llm_assistant import LLMAssistant

        assistant = LLMAssistant(
            planner_model="opencode/mimo-v2.5-free",
            coder_model="opencode/deepseek-v4-flash-free",
        )
        assert assistant.planner_model == "opencode/mimo-v2.5-free"
        assert assistant.coder_model == "opencode/deepseek-v4-flash-free"

    def test_opencode_model_prefix_detected(self):
        """El prefijo opencode/ se reconoce como modelo remoto."""
        assert "opencode/mimo-v2.5-free".startswith("opencode/")
