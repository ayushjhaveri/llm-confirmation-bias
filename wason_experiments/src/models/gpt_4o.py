"""
OpenAI-based wrapper (e.g., gpt-4o) with the same interface/params as Llama33_70B.

- Uses OpenAI chat.completions API (OpenAI Python SDK v1+).
- Expects a list of chat messages: [{"role": "user"/"assistant"/"system", "content": "..."}].
- Returns a raw text string (no usage dict), so it plugs into the same pipeline as vLLM wrappers.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Union, Optional
import os
import time
import random

from openai import OpenAI, APIError, RateLimitError  # pip install openai>=1.40.0

# Keep the parameter *structure* aligned with Llama33_70B
DEFAULT_OPENAI_STOPS = ["```"]  # harmless default; rarely triggered for chat models


@dataclass
class GenParams:
    # Same fields as your Llama GenParams; not all are sent to the API.
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 20          # kept for structural parity, not sent to API
    max_new_tokens: Optional[int] = 256
    repetition_penalty: Optional[float] = 1.0  # not sent to API
    stop: Optional[List[str]] = None


class OpenAIChat:
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        org_id: Optional[str] = None,
    ):
        """
        model: e.g. "gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", etc.
        """
        self.model_name = model

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Put it in your environment or .env.")

        org_id = org_id or os.getenv("OPENAI_ORG_ID")

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if org_id:
            client_kwargs["organization"] = org_id
        self.client = OpenAI(**client_kwargs)

        # Hardcoded decoding defaults (mirrors Llama33_70B keys)
        self._defaults: Dict[str, Any] = {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_new_tokens": 256,
            "repetition_penalty": 1.0,
            "stop": DEFAULT_OPENAI_STOPS,
        }

        # Remote OpenAI chat models are not "thinking models" in your local sense
        self.supports_think = False

    # --- internal helpers -------------------------------------------------

    def _effective_params(self, override: Optional[GenParams]) -> Dict[str, Any]:
        """
        Merge defaults with an optional GenParams override.
        This stays parallel to Llama33_70B._to_sampling_params, but we later
        drop unsupported keys when calling the OpenAI API.
        """
        d = dict(self._defaults)
        if override is not None:
            if override.temperature is not None:
                d["temperature"] = override.temperature
            if override.top_p is not None:
                d["top_p"] = override.top_p
            if override.top_k is not None:
                d["top_k"] = override.top_k
            if override.max_new_tokens is not None:
                d["max_new_tokens"] = override.max_new_tokens
            if override.repetition_penalty is not None:
                d["repetition_penalty"] = override.repetition_penalty
            if override.stop is not None:
                d["stop"] = override.stop
        return d

    def _build_api_kwargs(
        self,
        messages: List[Dict[str, str]],
        eff: Dict[str, Any],
        omit_temperature: bool = False,
    ) -> Dict[str, Any]:
        """
        Convert our internal params into OpenAI chat.completions kwargs.
        Only send keys that the API actually supports.
        """
        max_tokens = int(eff["max_new_tokens"])
        temperature = float(eff["temperature"])
        top_p = float(eff["top_p"])
        stop = eff["stop"]

        kw: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
        }

        # Newer families (gpt-4.1, gpt-5*, o1/o3/o4) want max_completion_tokens
        if self.model_name.startswith(("gpt-5", "gpt-4.1", "o1", "o3", "o4")):
            kw["max_completion_tokens"] = max_tokens
        else:
            kw["max_tokens"] = max_tokens

        # temperature: many newer models disallow temperature; we handle via omit_temperature flag
        if not omit_temperature:
            if not self.model_name.startswith(("gpt-5", "gpt-4.1", "o1", "o3", "o4")):
                kw["temperature"] = temperature

        # top_p is widely supported and can be passed safely
        kw["top_p"] = top_p

        # stop strings (if any)
        if stop:
            kw["stop"] = stop

        # NOTE: we intentionally do not send top_k or repetition_penalty since
        # OpenAI chat API does not support them; they are only kept for parity
        # with your local wrappers and for describe_params().
        return kw

    # --- public API -------------------------------------------------------

    def name(self) -> str:
        return f"openai/{self.model_name}"

    def generate(
        self,
        messages: Union[str, List[Dict[str, str]]],
        params: Optional[GenParams] = None,
    ) -> str:
        """
        Mirrors Llama33_70B.generate, but calls OpenAI chat.completions.
        - If `messages` is a list of dicts, it's passed through.
        - If `messages` is a string, it's wrapped as a single user message.
        Returns a plain string (assistant content).
        """
        if isinstance(messages, str):
            msg_list: List[Dict[str, str]] = [{"role": "user", "content": messages}]
        else:
            msg_list = messages

        eff = self._effective_params(params)
        last_err: Optional[Exception] = None

        for attempt in range(6):
            try:
                # First try with temperature included when allowed
                kwargs_api = self._build_api_kwargs(msg_list, eff, omit_temperature=False)
                resp = self.client.chat.completions.create(**kwargs_api)
                text = resp.choices[0].message.content or ""
                return text

            except (RateLimitError, APIError) as e:
                msg = getattr(e, "message", "") or str(e)

                # If 400 due to temperature, retry once without temperature in the payload
                if "Unsupported value: 'temperature'" in msg or "param': 'temperature'" in msg:
                    try:
                        kwargs_api = self._build_api_kwargs(msg_list, eff, omit_temperature=True)
                        resp = self.client.chat.completions.create(**kwargs_api)
                        text = resp.choices[0].message.content or ""
                        return text
                    except (RateLimitError, APIError) as e2:
                        last_err = e2
                else:
                    last_err = e

                # Exponential backoff
                sleep_s = min(2 ** attempt + random.random(), 20.0)
                time.sleep(sleep_s)
                if attempt == 5:
                    raise last_err  # type: ignore[operator]

        raise RuntimeError("OpenAI call failed after retries.")

    def describe_params(self) -> Dict[str, Any]:
        """
        Mirrors describe_params() from Llama33_70B, but marks backend='openai'.
        """
        d = self._defaults
        return {
            "model_name": self.model_name,
            "model_path": f"openai://{self.model_name}",
            "backend": "openai",
            "dtype": "n/a",
            "temperature": d["temperature"],
            "top_p": d["top_p"],
            "top_k": d["top_k"],
            "max_new_tokens": d["max_new_tokens"],
            "repetition_penalty": d["repetition_penalty"],
            "stop": d["stop"],
            # Static placeholders for compatibility with your local wrappers
            "tensor_parallel_size": None,
            "gpu_memory_utilization": None,
            "max_model_len": None,
        }
