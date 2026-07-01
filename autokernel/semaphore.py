"""Resource coordination for LLM inference and GPU benchmarks."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

import ollama as ollama_client

logger = logging.getLogger(__name__)


@dataclass
class ResourceStats:
    """Track resource usage metrics."""

    llm_switches: int = 0
    llm_inference_count: int = 0
    llm_inference_total_seconds: float = 0.0
    gpu_bench_count: int = 0
    gpu_bench_total_seconds: float = 0.0
    vram_peak_mb: float = 0.0


class ResourceSemaphore:
    """Coordinate LLM inference (VRAM) and GPU bench (compute).

    Rules:
    - Only 1 LLM model loaded at a time (VRAM constraint)
    - GPU bench is exclusive (no concurrent bench runs)
    - LLM and GPU bench CAN overlap (different resources)
    """

    def __init__(self):
        self.llm_sem = asyncio.Semaphore(1)
        self.gpu_sem = asyncio.Semaphore(1)
        self.current_model: str | None = None
        self.stats = ResourceStats()
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def llm(self, model: str):
        """Acquire LLM resource. Unloads previous model if switching."""
        async with self.llm_sem:
            t0 = time.monotonic()
            async with self._lock:
                if self.current_model != model:
                    await self._switch_model(model)
                self.stats.llm_inference_count += 1
            try:
                yield
            finally:
                elapsed = time.monotonic() - t0
                async with self._lock:
                    self.stats.llm_inference_total_seconds += elapsed

    async def _switch_model(self, model: str):
        """Stop current model and load new one."""
        if self.current_model:
            try:
                ollama_client.stop(self.current_model)
                await asyncio.sleep(0.5)
            except Exception as exc:
                logger.warning("ollama stop failed for %s: %s", self.current_model, exc)
            self.stats.llm_switches += 1

        # Pre-load model (non-blocking)
        try:
            ollama_client.generate(model=model, prompt="", keep_alive="5m")
        except Exception as exc:
            logger.warning("ollama pre-load failed for %s: %s", model, exc)
        self.current_model = model

    @asynccontextmanager
    async def gpu(self):
        """Acquire GPU compute resource (exclusive bench)."""
        async with self.gpu_sem:
            t0 = time.monotonic()
            self.stats.gpu_bench_count += 1
            try:
                yield
            finally:
                elapsed = time.monotonic() - t0
                self.stats.gpu_bench_total_seconds += elapsed

    def get_stats(self) -> dict:
        """Return resource usage stats."""
        return {
            "llm_model": self.current_model,
            "llm_switches": self.stats.llm_switches,
            "llm_inferences": self.stats.llm_inference_count,
            "llm_seconds": round(self.stats.llm_inference_total_seconds, 1),
            "gpu_bench_count": self.stats.gpu_bench_count,
            "gpu_seconds": round(self.stats.gpu_bench_total_seconds, 1),
        }

    async def cleanup(self):
        """Unload all models."""
        if self.current_model:
            try:
                ollama_client.stop(self.current_model)
            except Exception as exc:
                logger.warning("ollama cleanup failed for %s: %s", self.current_model, exc)
            self.current_model = None
