# lm_api.py
# (drop-in replacement for your current lm_api.py; keeps old API working)
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import backoff
import openai

# ---- local vLLM backend ----

@dataclass
class _UsageShim:
    prompt_tokens: int
    completion_tokens: int

@dataclass
class _MsgShim:
    content: str
    reasoning_content: Optional[str] = None  # kept for compatibility

@dataclass
class _ChoiceShim:
    message: _MsgShim

@dataclass
class _ChatCompletionShim:
    choices: List[_ChoiceShim]
    usage: object  # allow dict


class _LocalNonThinkingVLLM:
    """
    vLLM wrapper compatible with OpenAI chat.completions.create
    - apply_chat_template(add_generation_prompt=True) WITHOUT enable_thinking
    - NO '/think' suffix
    - supports any non-thinking instruct model (llama, mistral, etc.)
    """
    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.95,
        max_model_len: int = 8192,
        enforce_eager: bool = True,
        trust_remote_code: bool = True,
    ):
        from vllm import LLM  # local import

        stop_tokens = ["<|im_end|>", "<|endoftext|>", "</s>"]

        if "llama" in model_path.lower():
            tensor_parallel_size = 2
            stop_tokens = ["<|eot_id|>", "<|end_of_text|>", "</s>", "```"]


        self.model_path = model_path
        base = os.path.basename(model_path.rstrip("/"))
        self.model_name = base or "local-vllm"

        self._defaults = {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_new_tokens": 256,  # safer default for non-thinking
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "min_p": 0.0,
            "stop": stop_tokens,  # generic-ish
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
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            try:
                self._tokenizer = self.engine.get_tokenizer()
            except Exception:
                self._tokenizer = self.engine.llm_engine.tokenizer
        return self._tokenizer

    def _as_chat_prompt_from_messages(self, messages):
        tok = self._get_tokenizer()
        # IMPORTANT: no enable_thinking kwarg; works across tokenizers
        if "qwen" in self.model_path.lower():
            # Explicitly disable thinking for Qwen
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # <-- important
            )
        else:
            # Other models (llama, mistral, etc.)
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def generate_messages(self, messages, override: dict | None = None):
        from vllm import SamplingParams  # local import

        d = dict(self._defaults)
        if override:
            if "temperature" in override and override["temperature"] is not None:
                d["temperature"] = override["temperature"]
            if "top_p" in override and override["top_p"] is not None:
                d["top_p"] = override["top_p"]
            if "top_k" in override and override["top_k"] is not None:
                d["top_k"] = override["top_k"]
            if "max_tokens" in override and override["max_tokens"] is not None:
                d["max_new_tokens"] = override["max_tokens"]
            if "stop" in override and override["stop"] is not None:
                d["stop"] = override["stop"]
            if "repetition_penalty" in override and override["repetition_penalty"] is not None:
                d["repetition_penalty"] = override["repetition_penalty"]
            if "presence_penalty" in override and override["presence_penalty"] is not None:
                d["presence_penalty"] = override["presence_penalty"]
            if "min_p" in override and override["min_p"] is not None:
                d["min_p"] = override["min_p"]

        sp = SamplingParams(
            temperature=float(d["temperature"]),
            top_p=float(d["top_p"]),
            top_k=int(d["top_k"]),
            max_tokens=int(d["max_new_tokens"]),
            repetition_penalty=float(d["repetition_penalty"]),
            presence_penalty=float(d["presence_penalty"]),
            min_p=float(d["min_p"]),
            stop=d["stop"],
        )

        prompt = self._as_chat_prompt_from_messages(messages)
        outs = self.engine.generate([prompt], sp)
        text = outs[0].outputs[0].text if outs and outs[0].outputs else ""

        completion_tokens = len(outs[0].outputs[0].token_ids) if outs and outs[0].outputs else 0
        tok = self._get_tokenizer()
        prompt_tokens = len(tok(prompt).input_ids)

        return text.strip(), prompt_tokens, completion_tokens

    def generate(self, system_message: str, user_message: str, override: dict | None = None):
        msgs = []
        if system_message and system_message.strip():
            msgs.append({"role": "system", "content": system_message})
        msgs.append({"role": "user", "content": user_message})
        return self.generate_messages(msgs, override=override)


class _LocalQwenVLLM:
    """
    vLLM wrapper compatible with OpenAI chat.completions.create
    - apply_chat_template(enable_thinking=True)
    - appends '/think'
    - uses defaults similar to your thesis wrapper
    """
    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.95,
        max_model_len: int = 32768,
        enforce_eager: bool = True,
        trust_remote_code: bool = True,
    ):
        from vllm import LLM  # local import

        stop_tokens = ["<|im_end|>", "<|endoftext|>", "</s>"]

        if "llama" in model_path.lower():
            tensor_parallel_size = 2
            stop_tokens = ["<|eot_id|>", "<|end_of_text|>", "</s>", "```"]

        self.model_path = model_path
        base = os.path.basename(model_path.rstrip("/"))
        self.model_name = base or "qwen3-8b"

        self._defaults = {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_new_tokens": 16384,
            "repetition_penalty": 1.0,
            "presence_penalty": 1.0,
            "min_p": 0.0,
            "stop": stop_tokens,
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
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            try:
                self._tokenizer = self.engine.get_tokenizer()
            except Exception:
                self._tokenizer = self.engine.llm_engine.tokenizer
        return self._tokenizer

    def _as_chat_prompt_from_messages(self, messages):
        tok = self._get_tokenizer()
        # IMPORTANT: no enable_thinking kwarg; works across tokenizers
        if "qwen" in self.model_path.lower():
            # Explicitly disable thinking for Qwen
            prompt = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            return prompt + "/think"
        else:
            # Other models (llama, mistral, etc.)
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def generate_messages(
        self,
        messages: List[Dict[str, str]],
        override: dict | None = None,
    ) -> tuple[str, int, int]:
        from vllm import SamplingParams  # local import

        d = dict(self._defaults)
        if override:
            if "temperature" in override and override["temperature"] is not None:
                d["temperature"] = override["temperature"]
            if "top_p" in override and override["top_p"] is not None:
                d["top_p"] = override["top_p"]
            if "top_k" in override and override["top_k"] is not None:
                d["top_k"] = override["top_k"]
            # OpenAI compat mapping:
            if "max_tokens" in override and override["max_tokens"] is not None:
                d["max_new_tokens"] = override["max_tokens"]
            if "stop" in override and override["stop"] is not None:
                d["stop"] = override["stop"]

        sp = SamplingParams(
            temperature=float(d["temperature"]),
            top_p=float(d["top_p"]),
            top_k=int(d["top_k"]),
            max_tokens=int(d["max_new_tokens"]),
            repetition_penalty=float(d["repetition_penalty"]),
            presence_penalty=float(d["presence_penalty"]),
            min_p=float(d["min_p"]),
            stop=d["stop"],
        )

        prompt = self._as_chat_prompt_from_messages(messages)
        outs = self.engine.generate([prompt], sp)
        text = outs[0].outputs[0].text if outs and outs[0].outputs else ""

        completion_tokens = len(outs[0].outputs[0].token_ids) if outs and outs[0].outputs else 0
        tok = self._get_tokenizer()
        prompt_tokens = len(tok(prompt).input_ids)

        return text.strip(), prompt_tokens, completion_tokens

    # backward-compat: old callers use (system_message, user_message)
    def generate(
        self,
        system_message: str,
        user_message: str,
        override: dict | None = None,
    ) -> tuple[str, int, int]:
        msgs: List[Dict[str, str]] = []
        if system_message and system_message.strip():
            msgs.append({"role": "system", "content": system_message})
        msgs.append({"role": "user", "content": user_message})
        return self.generate_messages(msgs, override=override)


def get_client(model_name: str, thinking_mode: bool = True):
    """Initialize and return the API client based on the model type"""
    if model_name.startswith("gpt-"):
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = "https://api.openai.com/v1"
    elif model_name.startswith(("o1-", "o3-", "o4-")):
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = "https://api.openai.com/v1"
    elif model_name.startswith("deepseek-"):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = "https://api.deepseek.com"
    elif model_name.startswith("ollama/"):
        base_url = "http://localhost:11434/v1"
        api_key = "ollama"  # required but unused
    elif model_name.startswith("local_vllm:"):
        model_path = model_name.split("local_vllm:", 1)[1]
        if thinking_mode:
            return _LocalQwenVLLM(model_path=model_path)
        else:
            return _LocalNonThinkingVLLM(model_path=model_path)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return openai.OpenAI(api_key=api_key, base_url=base_url)


# NOTE: price per 1M token, recorded on 2025-03-23
API_PRICING = {
    # OpenAI models
    "gpt-4o-2024-05-13": {"input": 5.0, "output": 15.0},
    "gpt-4o-2024-08-06": {"input": 2.5, "output": 10.0},
    "gpt-4o-2024-11-20": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.6},
    "o1-2024-12-17": {"input": 15.0, "output": 60.0},
    "o1-pro-2025-03-19": {"input": 150.0, "output": 600.0},
    "o1-mini-2024-09-12": {"input": 1.10, "output": 4.40},
    "o3-mini-2025-01-31": {"input": 1.10, "output": 4.40},
    "o4-mini-2025-04-16": {"input": 1.10, "output": 4.40},
    # Deepseek
    "deepseek-chat": {"input": 0.27, "output": 1.10},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
}


def calculate_api_cost(prompt_tokens, completion_tokens, model) -> float:
    if model in API_PRICING:
        input_cost = (prompt_tokens / 1_000_000) * API_PRICING[model]["input"]
        output_cost = (completion_tokens / 1_000_000) * API_PRICING[model]["output"]
        return input_cost + output_cost
    warnings.warn(f"Unknown model: {model}. No pricing information.", UserWarning)
    return float("nan")


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def query_llm(
    client,
    model: str,
    system_message: Optional[str] = None,
    user_message: Optional[str] = None,
    chat_kwargs: dict = {},
    messages: Optional[List[Dict[str, str]]] = None,
) -> Tuple[object, float]:
    """
    New: pass `messages=[{"role":"user","content":"..."}, ...]` to use true chat history.
    Backward compat: pass system_message + user_message.
    """
    if messages is None:
        assert user_message is not None, "Provide user_message or messages"
        messages = []
        if system_message and system_message.strip():
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})

    if model.startswith("local_vllm:"):
        text, ptok, ctok = client.generate_messages(messages, override=chat_kwargs)
        response = _ChatCompletionShim(
            choices=[_ChoiceShim(message=_MsgShim(content=text, reasoning_content=None))],
            usage={"prompt_tokens": int(ptok), "completion_tokens": int(ctok)},
        )
        return response, 0.0

    if model.startswith("ollama/"):
        model_name = model.split("ollama/")[1]
    else:
        model_name = model

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        **chat_kwargs,
    )

    if model.startswith("ollama/"):
        cost = 0.0
    else:
        cost = calculate_api_cost(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            model=model_name,
        )
    return response, cost
