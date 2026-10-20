"""Token accounting across LLM calls: OpenAI-style `usage` and llama.cpp `timings`."""

from typing import Any
from dataclasses import asdict, dataclass

@dataclass
class TokenCounter:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0  # prompt tokens served from the KV cache
    prefill_s: float = 0.0
    decode_s: float = 0.0

    def add(self, usage: dict[str, Any] | None = None, timings: dict[str, Any] | None = None) -> None:
        """Count one call. llama.cpp `timings` (prompt_n = new tokens, cache_n = cached) take priority for
        the prompt split; OpenAI `usage` fills in whatever timings do not have."""
        usage = usage or {}
        timings = timings or {}
        self.calls += 1
        if "prompt_n" in timings:
            new, cached = int(timings.get("prompt_n") or 0), int(timings.get("cache_n") or 0)
            self.prompt_tokens += new + cached
            self.cached_tokens += cached
        else:
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            details = usage.get("prompt_tokens_details") or {}
            self.cached_tokens += int(details.get("cached_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or timings.get("predicted_n") or 0)
        self.prefill_s += float(timings.get("prompt_ms") or 0) / 1000
        self.decode_s += float(timings.get("predicted_ms") or 0) / 1000

    @property
    def cached_share(self) -> float | None:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else None

    @property
    def decode_tps(self) -> float | None:
        return self.completion_tokens / self.decode_s if self.decode_s else None

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "cached_share": self.cached_share, "decode_tps": self.decode_tps}
