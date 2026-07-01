"""Multi-model LLM router for AutoKernel pipeline.

Models:
  - ornith:9b       → planning, spec generation, NCU analysis (local Ollama)
  - qwen2.5-coder:7b → kernel coding, test generation (local Ollama)
  - nemotron-3-ultra → architectural review, CUDA migration (NVIDIA API)
  - bge-m3          → embeddings for RAG (local Ollama)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import ollama as ollama_client

try:
    import aiohttp
except ImportError:
    aiohttp = None

from autokernel.nemotron_client import NemotronClient, ReviewResult
from autokernel.rag_index import RAGIndex

SCRIPT_DIR = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SPEC_PROMPT = """You are generating a Triton kernel optimization spec.

Kernel type: {kernel_type}
Profile data:
{profile_data}

RAG context (Triton docs + similar kernels):
{context}

Generate a complete optimization spec with:
1. **Goal**: What the kernel does, target throughput, peak % goal
2. **Current bottlenecks**: From profile data (memory bandwidth, compute, latency)
3. **Optimization strategies**: Ranked list with expected impact
4. **Config parameters**: BLOCK_SIZE_K, GROUP_SIZE_M, stages, etc.
5. **Input shapes**: Test cases from profile data
6. **Acceptance criteria**: min speedup, correctness threshold

Output as JSON with keys: goal, bottlenecks, strategies, config, test_shapes, acceptance
"""

TEST_GEN_PROMPT = """Generate pytest tests for a Triton kernel.

Spec:
{spec}

Kernel type: {kernel_type}

Generate complete pytest tests that:
1. Test correctness against PyTorch reference for each input shape
2. Test edge cases (empty, single element, power-of-2 sizes)
3. Test dtypes (fp16, bf16, f32 where applicable)
4. Use pytest.mark.parametrize for shapes
5. Assert torch.allclose with appropriate tolerances

Output only the test code, no explanation.
"""

KERNEL_GEN_PROMPT = """Implement a Triton kernel based on this spec and tests.

Spec:
{spec}

Tests (must pass):
{tests}

Kernel type: {kernel_type}

Requirements:
1. Use @triton.jit decorator
2. Include autotuning if beneficial
3. Handle multiple dtypes
4. Follow existing kernel conventions in the codebase
5. Include docstring with memory/compute analysis

Output only the kernel code, no explanation.
"""

NCU_ANALYSIS_PROMPT = """Analyze this NCU profiling log and suggest optimizations.

Kernel type: {kernel_type}
NCU log:
{ncu_log}

Identify:
1. **Bottleneck**: memory-bound vs compute-bound, with evidence
2. **Occupancy**: current vs theoretical max
3. **Memory**: cache hit rate, bank conflicts, uncoalesced accesses
4. **Compute**: tensor core utilization, warp divergence
5. **Config suggestions**: specific BLOCK_SIZE, stages, GROUP_SIZE changes

Output as JSON with keys: bottleneck, occupancy, memory_issues, compute_issues, config_changes
"""

CUDA_MIGRATION_PROMPT = """Migrate this Triton kernel to CUDA C++ for Blackwell SM 12.0.

Kernel type: {kernel_type}
Target arch: {target_arch}

Triton kernel:
```python
{triton_kernel}
```

Spec:
{spec}

Requirements:
1. Complete, compilable CUDA C++ code with pybind11 bindings
2. Use CUTLASS for matmul kernels if applicable
3. Use wmma/mma.sync for tensor cores on SM 12.0
4. Include proper shared memory management
5. Handle FP16/BF16/FP32 dtypes
6. Include a Python wrapper class that matches PyTorch's interface

Output complete CUDA code with headers and Python bindings.
"""


@dataclass
class Spec:
    """Optimization specification for a kernel."""

    kernel_type: str
    goal: str = ""
    bottlenecks: list[str] = field(default_factory=list)
    strategies: list[dict] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    test_shapes: list[dict] = field(default_factory=list)
    acceptance: dict[str, Any] = field(default_factory=dict)
    raw_json: str = ""

    @classmethod
    def from_json(cls, kernel_type: str, raw: str) -> Spec:
        """Parse JSON spec from LLM output."""
        # Extract JSON from markdown code block if present
        json_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", raw, re.DOTALL)
        json_str = json_match.group(1) if json_match else raw

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Try to find JSON object in text
            obj_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if obj_match:
                data = json.loads(obj_match.group())
            else:
                return cls(kernel_type=kernel_type, raw_json=raw)

        return cls(
            kernel_type=kernel_type,
            goal=data.get("goal", ""),
            bottlenecks=data.get("bottlenecks", []),
            strategies=data.get("strategies", []),
            config=data.get("config", {}),
            test_shapes=data.get("test_shapes", []),
            acceptance=data.get("acceptance", {}),
            raw_json=raw,
        )


class LLMAssistant:
    """Multi-model LLM assistant with RAG for kernel optimization."""

    def __init__(
        self,
        planner_model: str = "ornith:9b",
        coder_model: str = "qwen2.5-coder:7b",
        embed_model: str = "bge-m3",
    ):
        self.planner_model = planner_model
        self.coder_model = coder_model
        self.nemotron = NemotronClient()
        self.rag = RAGIndex(embed_model)
        self._current_model: str | None = None

    def _read_openrouter_key(self) -> str:
        """Read OpenRouter API key from env or Proton Pass."""
        env_key = os.environ.get("OPENROUTER_API_KEY")
        if env_key:
            return env_key
        try:
            result = subprocess.run(
                [
                    "/home/alexendros/.local/bin/pass-cli",
                    "item",
                    "view",
                    "pass://Infraestructura/OpenRouter/APIkey",
                    "--field",
                    "APIkey",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            return result.stdout.strip()
        except Exception:
            raise RuntimeError(
                "OPENROUTER_API_KEY not found in environment or Proton Pass"
            )

    def _call_api(self, model: str, messages: list[dict[str, str]]) -> str:
        """Call remote API (OpenRouter) for opencode models."""
        import urllib.request

        api_key = self._read_openrouter_key()
        payload = json.dumps(
            {
                "model": model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 8192,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")[:500]
            raise RuntimeError(f"OpenRouter {e.code}: {body}")
        return data["choices"][0]["message"]["content"]

    def _call_ollama(self, model: str, prompt: str, system: str = "") -> str:
        """Call Ollama or remote API depending on model prefix."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if model.startswith("opencode/"):
            return self._call_api(model, messages)

        if self._current_model != model:
            self._switch_model(model)

        resp = ollama_client.chat(
            model=model, messages=messages, options={"temperature": 0.1}
        )
        return resp["message"]["content"]

    def _switch_model(self, model: str):
        """Switch active model (unloads previous to save VRAM)."""
        if self._current_model:
            try:
                ollama_client.stop(self._current_model)
            except Exception:
                pass
        self._current_model = model

    # ------------------------------------------------------------------
    # Planning (ornith:9b + RAG)
    # ------------------------------------------------------------------

    def generate_spec(self, kernel_type: str, profile_data: dict) -> Spec:
        """Generate optimization spec using planner model + RAG context."""
        context = self.rag.query(
            f"Triton {kernel_type} optimization Blackwell SM12.0", k=5
        )
        context_text = "\n\n".join(
            f"--- {c.source} (score={c.metadata.get('score', 0):.2f}) ---\n{c.text}"
            for c in context
        )

        prompt = SPEC_PROMPT.format(
            kernel_type=kernel_type,
            profile_data=json.dumps(profile_data, indent=2),
            context=context_text,
        )
        raw = self._call_ollama(
            self.planner_model,
            prompt,
            system="You are a Triton GPU kernel optimization expert. Output structured JSON.",
        )
        return Spec.from_json(kernel_type, raw)

    def analyze_ncu(self, kernel_type: str, ncu_log: str) -> dict:
        """Analyze NCU log and suggest config changes."""
        prompt = NCU_ANALYSIS_PROMPT.format(kernel_type=kernel_type, ncu_log=ncu_log)
        raw = self._call_ollama(
            self.planner_model,
            prompt,
            system="You are an NCU profiling expert. Output structured JSON.",
        )
        json_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", raw, re.DOTALL)
        json_str = json_match.group(1) if json_match else raw
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"error": "Failed to parse NCU analysis", "raw": raw}

    # ------------------------------------------------------------------
    # Coding (qwen2.5-coder:7b)
    # ------------------------------------------------------------------

    def generate_tests(self, spec: Spec) -> str:
        """Generate pytest tests from spec (TDD)."""
        prompt = TEST_GEN_PROMPT.format(
            spec=json.dumps(spec.__dict__, indent=2, default=str),
            kernel_type=spec.kernel_type,
        )
        raw = self._call_ollama(
            self.coder_model,
            prompt,
            system="You are a GPU kernel test engineer. Output only Python test code. No markdown, no code blocks, no explanations.",
        )
        # Strip markdown code blocks if present
        code = raw.strip()
        if code.startswith("```"):
            first_newline = code.index("\n")
            code = code[first_newline + 1 :]
        if code.endswith("```"):
            code = code[:-3]
        return code.strip()

    def generate_kernel(self, spec: Spec, tests: str) -> str:
        """Generate kernel implementation from spec + tests."""
        prompt = KERNEL_GEN_PROMPT.format(
            spec=json.dumps(spec.__dict__, indent=2, default=str),
            tests=tests,
            kernel_type=spec.kernel_type,
        )
        raw = self._call_ollama(
            self.coder_model,
            prompt,
            system="You are a Triton kernel developer. Output only Python kernel code. No markdown, no code blocks, no explanations.",
        )
        # Strip markdown code blocks if present
        code = raw.strip()
        if code.startswith("```"):
            # Remove opening block (```python or ```)
            first_newline = code.index("\n")
            code = code[first_newline + 1 :]
        if code.endswith("```"):
            code = code[:-3]
        return code.strip()

    # ------------------------------------------------------------------
    # Review (Nemotron API)
    # ------------------------------------------------------------------

    async def review_kernel(
        self,
        kernel_code: str,
        kernel_type: str,
        spec: str = "",
        bench_results: str = "",
        ncu_log: str = "",
    ) -> ReviewResult:
        """Review kernel using Nemotron-3-Ultra API."""
        return await self.nemotron.review(
            kernel_code=kernel_code,
            kernel_type=kernel_type,
            spec=spec,
            bench_results=bench_results,
            ncu_log=ncu_log,
        )

    # ------------------------------------------------------------------
    # CUDA Migration (Nemotron API)
    # ------------------------------------------------------------------

    async def migrate_to_cuda(
        self,
        triton_kernel: str,
        kernel_type: str,
        spec: str = "",
    ) -> str:
        """Migrate Triton kernel to CUDA C++ via Nemotron."""
        return await self.nemotron.generate_cuda(
            triton_kernel=triton_kernel,
            kernel_type=kernel_type,
            spec=spec,
        )
