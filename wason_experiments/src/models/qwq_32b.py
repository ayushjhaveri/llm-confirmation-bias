"""
vLLM-based wrapper for local QwQ 32B (Qwen-family, reasoning).
- Hardcoded decoding defaults (overridable via optional params like Llama)
- Uses tokenizer.apply_chat_template on a list of chat turns
- No engine config introspection in describe_params()
- Treats QwQ as a "thinking" model; do NOT strip <think> here (leave raw to caller)
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Union, Optional
import os
from vllm import LLM, SamplingParams

DEFAULT_QWQ_STOPS = ["<|im_end|>", "<|endoftext|>", "</s>"]
CONVO_STOPS: List[str] = []  # Intentionally empty per your preference


@dataclass
class GenParamsCompat:
    """
    Optional override parameters to mirror the Llama wrapper behavior.
    Any field left as None will fall back to the model defaults.
    """
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_new_tokens: Optional[int] = None
    repetition_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    min_p: Optional[float] = None
    stop: Optional[List[str]] = None


class QwQ_32B:
    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        tensor_parallel_size: Optional[int] = None,
        gpu_memory_utilization: float = 0.95,
        max_model_len: int = 32768,
        enforce_eager: bool = True,
        trust_remote_code: bool = True,
    ):
        self.model_path = model_path
        base = os.path.basename(model_path.rstrip("/"))
        self.model_name = base or "qwq-32b"

        if tensor_parallel_size is None:
            tensor_parallel_size = int(os.environ.get("TP_SIZE", "1"))

        # Defaults (can be overridden per-call via GenParamsCompat)
        self._defaults: Dict[str, Any] = {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,           # keep consistent with your latest snippet
            "max_new_tokens": 16384,
            "repetition_penalty": 1.0,
            "presence_penalty": 1.0,
            "min_p": 0.0,
            "stop": DEFAULT_QWQ_STOPS + CONVO_STOPS,
            "max_model_len": 32768
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
        self._tokenizer = None
        self.supports_think = True

    def _get_tokenizer(self):
        if self._tokenizer is None:
            try:
                self._tokenizer = self.engine.get_tokenizer()
            except Exception:
                self._tokenizer = self.engine.llm_engine.tokenizer
        return self._tokenizer

    def _as_chat_prompt(self, prompt: Union[str, List[Dict[str, str]]]) -> str:
        tok = self._get_tokenizer()
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": str(prompt)}]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _to_sampling_params(self, override: Optional[GenParamsCompat] = None) -> SamplingParams:
        # start from hardcoded defaults
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
            if override.presence_penalty is not None:
                d["presence_penalty"] = override.presence_penalty
            if override.min_p is not None:
                d["min_p"] = override.min_p
            if override.stop is not None:
                d["stop"] = override.stop

        return SamplingParams(
            temperature=d["temperature"],
            top_p=d["top_p"],
            top_k=d["top_k"],
            max_tokens=d["max_new_tokens"],
            repetition_penalty=d["repetition_penalty"],
            presence_penalty=d["presence_penalty"],
            min_p=d["min_p"],
            stop=d["stop"],
        )

    def generate(self, prompt: Union[str, List[Dict[str, str]]], params: Optional[GenParamsCompat] = None) -> str:
        sp = self._to_sampling_params(params)
        chat_formatted = self._as_chat_prompt(prompt)
        outs = self.engine.generate([chat_formatted], sp)
        text = outs[0].outputs[0].text if outs and outs[0].outputs else ""
        return text.strip()

    # No engine config introspection
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
            "presence_penalty": d["presence_penalty"],
            "min_p": d["min_p"],
            "stop": d["stop"],
            "supports_think": True,
            # Static placeholders only
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": None,
            "max_model_len": d["max_model_len"],
        }
