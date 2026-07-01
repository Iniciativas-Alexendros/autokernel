import os
import sys
from unittest import mock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autokernel.semaphore import ResourceSemaphore


@pytest.fixture
def semaphore():
    with mock.patch("autokernel.semaphore.ollama_client") as fake_ollama:
        fake_ollama.generate = mock.Mock()
        fake_ollama.stop = mock.Mock()
        yield ResourceSemaphore(), fake_ollama


class TestResourceSemaphore:
    """Tests para ResourceSemaphore."""

    @pytest.mark.anyio
    async def test_llm_acquire_counts(self, semaphore):
        sem, fake_ollama = semaphore
        async with sem.llm("model-a"):
            pass
        stats = sem.get_stats()
        assert stats["llm_inferences"] == 1
        assert stats["llm_model"] == "model-a"
        fake_ollama.generate.assert_called_once()

    @pytest.mark.anyio
    async def test_llm_switch_unloads_previous(self, semaphore):
        sem, fake_ollama = semaphore
        async with sem.llm("model-a"):
            pass
        async with sem.llm("model-b"):
            pass
        assert fake_ollama.stop.call_count == 1
        assert sem.get_stats()["llm_switches"] == 1

    @pytest.mark.anyio
    async def test_llm_stop_failure_is_logged(self, semaphore):
        sem, fake_ollama = semaphore
        fake_ollama.stop.side_effect = RuntimeError("stop failed")
        async with sem.llm("model-a"):
            pass
        async with sem.llm("model-b"):
            pass
        fake_ollama.stop.assert_called_once()

    @pytest.mark.anyio
    async def test_gpu_acquire_counts(self, semaphore):
        sem, _ = semaphore
        async with sem.gpu():
            pass
        stats = sem.get_stats()
        assert stats["gpu_bench_count"] == 1
        assert stats["gpu_seconds"] >= 0.0

    @pytest.mark.anyio
    async def test_cleanup_unloads_model(self, semaphore):
        sem, fake_ollama = semaphore
        async with sem.llm("model-a"):
            pass
        await sem.cleanup()
        assert sem.current_model is None
        fake_ollama.stop.assert_called()
