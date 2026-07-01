"""Nemotron-3-Ultra API client with Proton Pass secret management."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import Optional

import aiohttp

NEMOTRON_API_URL = "https://integrate.api.nvidia.com/v1"
NEMOTRON_MODEL = "nvidia/nemotron-3-ultra"


def _read_proton_pass(vault: str, item: str, field: str = "password") -> str:
    """Read a secret from Proton Pass via pass-cli.

    URI format: pass://<vault>/<item>/<field>
    """
    uri = f"pass://{vault}/{item}/{field}"
    try:
        result = subprocess.run(
            [
                "/home/alexendros/.local/bin/pass-cli",
                "item",
                "view",
                uri,
                "--field",
                field,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        # pass-cli not installed or failed, try env fallback
        env_key = "NEMOTRON_API_KEY"
        if env_key in os.environ:
            return os.environ[env_key]
        raise RuntimeError(
            f"Failed to read {uri} from Proton Pass and {env_key} not in environment"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to read {uri}: {e.stderr}")


@dataclass
class ReviewResult:
    """Structured review output from Nemotron."""

    approved: bool
    critical_issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    fixes: dict[str, str] = field(default_factory=dict)
    raw_response: str = ""

    @classmethod
    def parse(cls, raw: str) -> ReviewResult:
        """Parse Nemotron markdown review into structured result."""
        lines = raw.strip().split("\n")
        issues: list[str] = []
        warnings: list[str] = []
        suggestions: list[str] = []
        approved = True

        current_section = None
        for line in lines:
            lower = line.lower().strip()
            if "critical" in lower or "blocker" in lower:
                current_section = "critical"
                approved = False
            elif "warning" in lower:
                current_section = "warning"
            elif "suggestion" in lower or "improvement" in lower:
                current_section = "suggestion"
            elif "approved" in lower or "verdict" in lower:
                if "approve" in lower:
                    approved = True
                elif "reject" in lower or "fail" in lower:
                    approved = False

            if current_section == "critical" and line.strip().startswith("-"):
                issues.append(line.strip().lstrip("- "))
            elif current_section == "warning" and line.strip().startswith("-"):
                warnings.append(line.strip().lstrip("- "))
            elif current_section == "suggestion" and line.strip().startswith("-"):
                suggestions.append(line.strip().lstrip("- "))

        return cls(
            approved=approved,
            critical_issues=issues,
            warnings=warnings,
            suggestions=suggestions,
            raw_response=raw,
        )


class NemotronClient:
    """Async client for Nemotron-3-Ultra via NVIDIA API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or _read_proton_pass("Infraestructura", "NVIDIA", "APIkey")
        self.base_url = NEMOTRON_API_URL
        self.model = NEMOTRON_MODEL

    async def _request(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 8192,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"Nemotron API {resp.status}: {body[:500]}")
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

    async def review(
        self,
        kernel_code: str,
        kernel_type: str,
        spec: str = "",
        bench_results: str = "",
        ncu_log: str = "",
    ) -> ReviewResult:
        """Review kernel for correctness, performance, memory, race conditions."""
        from autokernel.prompts.review import REVIEW_PROMPT

        prompt = REVIEW_PROMPT.format(
            kernel_type=kernel_type,
            kernel_code=kernel_code,
            spec=spec,
            bench_results=bench_results,
            ncu_log=ncu_log,
        )
        messages = [
            {
                "role": "system",
                "content": "You are a CUDA/Triton GPU kernel expert reviewer. Be strict and precise.",
            },
            {"role": "user", "content": prompt},
        ]
        raw = await self._request(messages, temperature=0.1)
        return ReviewResult.parse(raw)

    async def generate_cuda(
        self,
        triton_kernel: str,
        kernel_type: str,
        spec: str = "",
        target_arch: str = "sm_120",
    ) -> str:
        """Migrate Triton kernel to CUDA C++."""
        from autokernel.prompts.cuda_migration import CUDA_MIGRATION_PROMPT

        prompt = CUDA_MIGRATION_PROMPT.format(
            triton_kernel=triton_kernel,
            kernel_type=kernel_type,
            spec=spec,
            target_arch=target_arch,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a CUDA C++ expert. Generate complete, compilable CUDA kernels. "
                    "Include all headers, kernel launch wrappers, and Python bindings. "
                    "Target Blackwell SM 12.0 architecture with PTX 8.x support."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        return await self._request(messages, temperature=0.1, max_tokens=16384)
