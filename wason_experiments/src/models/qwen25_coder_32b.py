"""
vLLM-based wrapper for local Qwen2.5-Coder-32B-Instruct.

Notes
- Tailored for SHORT, STRICT code-gen (Announce → Python function).
- Slightly cooler temperature, modest rep-penalty, longer max_new_tokens.
- Uses tokenizer.apply_chat_template on a list of chat turns or a single string.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Union, Optional
from typing import cast
import os

from vllm import LLM, SamplingParams

# Common Qwen stops found in chat/code templates.
DEFAULT_QWEN_STOPS = ["<|im_end|>", "<|endoftext|>", "```"]

class Qwen25Coder32B:
    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.98,
        max_model_len: int = 16384,   # wide headroom for code prompts
        enforce_eager: bool = True,
        trust_remote_code: bool = True,  # Qwen often requires this
    ):
        self.model_path = model_path
        base = os.path.basename(model_path.rstrip("/"))
        self.model_name = base or "qwen2.5-coder-32b-instruct"

        # Defaults tuned for CODE FUNCTION generation (short, deterministic)
        self._defaults: Dict[str, Any] = {
            "temperature": 0.15,
            "top_p": 0.95,
            # top_k = 0 means disabled in many stacks; vLLM expects int >= 0
            "top_k": 0,
            "max_new_tokens": 640,      # allow room for full def + nested helpers
            "repetition_penalty": 1.05, # nudge away from loops
            "stop": DEFAULT_QWEN_STOPS,
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

        self.supports_think = False
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            try:
                self._tokenizer = self.engine.get_tokenizer()
            except Exception:
                # Older vLLM attr
                self._tokenizer = self.engine.llm_engine.tokenizer
        return self._tokenizer

    def _as_chat_prompt(self, prompt: Union[str, List[Dict[str, str]]]) -> str:
        tok = self._get_tokenizer()
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": str(prompt)}]
        # Qwen chat template supports system/user/assistant; generation prompt needed.
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _to_sampling_params(self, override: Optional["GenParams"] = None) -> SamplingParams:
        d = dict(self._defaults)
        if override is not None:
            d["temperature"] = override.temperature if override.temperature is not None else d["temperature"]
            d["top_p"] = override.top_p if override.top_p is not None else d["top_p"]
            d["top_k"] = override.top_k if override.top_k is not None else d["top_k"]
            d["max_new_tokens"] = override.max_new_tokens if override.max_new_tokens is not None else d["max_new_tokens"]
            d["repetition_penalty"] = override.repetition_penalty if override.repetition_penalty is not None else d["repetition_penalty"]
            if getattr(override, "stop", None) is not None:
                d["stop"] = override.stop

        return SamplingParams(
            temperature=cast(float, d["temperature"]),
            top_p=cast(float, d["top_p"]),
            top_k=cast(int, d["top_k"]),
            max_tokens=cast(int, d["max_new_tokens"]),
            repetition_penalty=cast(float, d["repetition_penalty"]),
            stop=d["stop"],
        )

    def generate(self, prompt: Union[str, List[Dict[str, str]]], params: Optional["GenParams"] = None) -> str:
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
            # static metadata (no engine introspection here)
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": None,
            "max_model_len": None,
        }

# Match the simple GenParams API your other code expects.
@dataclass
class GenParams:
    temperature: float = 0.15
    top_p: float = 0.95
    top_k: int = 0
    max_new_tokens: int = 640
    repetition_penalty: float = 1.05
    stop: Optional[List[str]] = None
