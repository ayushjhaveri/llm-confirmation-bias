"""
vLLM-based wrapper for local Qwen3 8B Instruct (no thinking).
- Hardcoded decoding defaults (optionally overridable via GenParams)
- Uses tokenizer.apply_chat_template on a list of chat turns
- Does NOT enable thinking mode / <think> blocks
- No engine config introspection in describe_params()
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Union, Optional
import os
from vllm import LLM, SamplingParams
from typing import cast

# Qwen-style stop tokens; keep conversation markers out.
DEFAULT_QWEN_STOPS = ["<|im_end|>", "<|endoftext|>", "</s>", "```"]


@dataclass
class GenParams:
    """
    Optional override parameters (same pattern as Llama33_70B.GenParams).
    If a field is None, the default from this wrapper is used.
    """
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 20
    max_new_tokens: Optional[int] = 256
    repetition_penalty: Optional[float] = 1.0
    stop: Optional[List[str]] = None


class Qwen3_8B_NoThink:
    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.95,
        max_model_len: int = 8192,   # allow long context, like your thinking wrapper
        enforce_eager: bool = True,
        trust_remote_code: bool = True,
    ):
        self.model_path = model_path
        base = os.path.basename(model_path.rstrip("/"))
        # Fallback name if the folder name is weird
        self.model_name = base or "qwen3-8b-no-think"

        # Hardcoded decoding defaults (Llama-like: short max_new_tokens etc.)
        self._defaults: Dict[str, Any] = {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_new_tokens": 256,
            "repetition_penalty": 1.0,
            "stop": DEFAULT_QWEN_STOPS,
            "max_model_len": max_model_len,
        }

        self.engine = LLM(
            model=self.model_path,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            trust_remote_code=trust_remote_code,
        )

        # This is explicitly a non-thinking model wrapper
        self.supports_think = False
        self._tokenizer = None

    # ----------------------------------------------------------------------
    # Tokenizer / chat template helpers
    # ----------------------------------------------------------------------

    def _get_tokenizer(self):
        if self._tokenizer is None:
            try:
                self._tokenizer = self.engine.get_tokenizer()
            except Exception:
                # Fallback for older vLLM internals
                self._tokenizer = self.engine.llm_engine.tokenizer
        return self._tokenizer

    def _as_chat_prompt(self, prompt: Union[str, List[Dict[str, str]]]) -> str:
        """
        Standard chat template usage, WITHOUT thinking mode.
        """
        tok = self._get_tokenizer()
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": str(prompt)}]
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    # ----------------------------------------------------------------------
    # Sampling params
    # ----------------------------------------------------------------------

    def _to_sampling_params(self, override: Optional[GenParams] = None) -> SamplingParams:
        # start from hardcoded defaults
        d = dict(self._defaults)
        if override is not None:
            # only override if provided (keep None = use default)
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
                d["stop"] = override.stop  # allow caller to change stops

        return SamplingParams(
            temperature=cast(float, d["temperature"]),
            top_p=cast(float, d["top_p"]),
            top_k=cast(int, d["top_k"]),
            max_tokens=cast(int, d["max_new_tokens"]),
            repetition_penalty=cast(float, d["repetition_penalty"]),
            stop=d["stop"],
        )

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def generate(self, prompt: Union[str, List[Dict[str, str]]], params: Optional[GenParams] = None) -> str:
        sp = self._to_sampling_params(params)
        chat_formatted = self._as_chat_prompt(prompt)
        outs = self.engine.generate([chat_formatted], sp)
        text = outs[0].outputs[0].text if outs and outs[0].outputs else ""
        return text.strip()

    def describe_params(self) -> Dict[str, Any]:
        d = self._defaults
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "backend": "vllm",
            "dtype": "bfloat16",
            "temperature": d["temperature"],
            "top_p": d["top_p"],
            "top_k": d["top_k"],
            "max_new_tokens": d["max_new_tokens"],
            "repetition_penalty": d["repetition_penalty"],
            "stop": d["stop"],
            "supports_think": False,
            # Static metadata only (no engine introspection)
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": None,
            "max_model_len": None,
        }
