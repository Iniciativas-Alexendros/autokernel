import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

try:
    import requests
except ModuleNotFoundError:
    requests = None

import time


OLLAMA_BASE = "http://localhost:11434"


@pytest.fixture(autouse=True)
def _requires_requests():
    if requests is None:
        pytest.skip("requests no disponible (instalar con uv sync --extra testing)")


class TestOllamaHealth:
    """Tests de salud de Ollama."""

    def test_ollama_responds(self):
        """Ollama responde en puerto 11434."""
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        assert resp.status_code == 200

    def test_ornith_model_available(self):
        """Modelo ornith:9b esta disponible."""
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        assert any("ornith" in m for m in models), f"Modelos disponibles: {models}"

    def test_qwen_model_available(self):
        """Modelo qwen2.5-coder:7b sigue disponible."""
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        assert any("qwen2.5-coder" in m for m in models), (
            f"Modelos disponibles: {models}"
        )

    def test_model_loaded(self):
        """Al menos un modelo esta cargado en VRAM."""
        resp = requests.get(f"{OLLAMA_BASE}/api/ps", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data.get("models", [])) > 0


class TestOllamaGenerate:
    """Tests de generacion de codigo CUDA."""

    def _get_loaded_model(self) -> str:
        """Get the name of the currently loaded model."""
        resp = requests.get(f"{OLLAMA_BASE}/api/ps", timeout=5)
        models = resp.json().get("models", [])
        return models[0]["name"] if models else ""

    def test_generate_cuda_context(self):
        """Ollama genera completado para codigo CUDA."""
        model = self._get_loaded_model()
        if not model:
            pytest.skip("No hay modelo cargado en VRAM")
        prompt = "Complete this CUDA kernel for vector addition:\n__global__ void vec_add(float* a, float* b, float* c, int n) {"
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert len(data["response"]) > 0

    def test_generate_response_time(self):
        """Respuesta en <20 segundos para prompt corto."""
        model = self._get_loaded_model()
        if not model:
            pytest.skip("No hay modelo cargado en VRAM")
        prompt = "__global__ void kernel(float* x) { int idx = "
        start = time.time()
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=30,
        )
        elapsed = time.time() - start
        assert resp.status_code == 200
        assert elapsed < 20, f"Respuesta tardo {elapsed:.1f}s (>20s)"


class TestRightNowConfig:
    """Tests de configuracion de RightNow."""

    CONFIG_PATH = "/home/alexendros/.config/RightNowEditor/.rightnowrules"

    def test_rightnow_config_exists(self):
        """RightNow config (.rightnowrules) existe."""
        import os

        assert os.path.exists(self.CONFIG_PATH), f"No existe: {self.CONFIG_PATH}"

    def test_rightnow_config_has_ollama(self):
        """Config apunta a Ollama."""
        import json

        with open(self.CONFIG_PATH) as f:
            config = json.load(f)
        assert config.get("ai_provider") == "ollama"
        assert "localhost:11434" in config.get("ollama_url", "")

    def test_rightnow_config_uses_ornith(self):
        """Config usa ornith como modelo."""
        import json

        with open(self.CONFIG_PATH) as f:
            config = json.load(f)
        assert "ornith" in config.get("model", "")
