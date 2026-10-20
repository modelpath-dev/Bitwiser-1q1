"""Universal prompt template handler — Gemma / Llama / Mistral / Phi / Qwen."""
from __future__ import annotations

# Each entry maps a family-name pattern to a (system, user, assistant) wrapper.
# The wrapper is applied as: system_pre + system + system_post + user_pre + user + user_post + assistant_pre


_TEMPLATES: dict[str, dict] = {
    "gemma": {
        "user": "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n",
        "stop": ["<end_of_turn>"],
    },
    "llama": {
        # Llama-3 chat. No <|begin_of_text|>: llama-completion adds BOS itself,
        # and a second one makes it warn "the final prompt starts with 2 BOS tokens".
        "user": (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        "stop": ["<|eot_id|>", "<|end_of_text|>"],
    },
    "mistral": {
        # llama-cli auto-prepends BOS; we must NOT include <s> ourselves,
        # otherwise BOS appears twice and confuses the model.
        "user": "[INST] {prompt} [/INST]",
        "stop": ["</s>"],
    },
    "phi": {
        "user": "<|user|>\n{prompt}<|end|>\n<|assistant|>\n",
        "stop": ["<|end|>", "<|endoftext|>"],
    },
    "qwen": {
        "user": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        "stop": ["<|im_end|>"],
    },
}

_DEFAULT = {
    "user": "{prompt}",
    "stop": [],
}


def _family_for(name_or_arch: str) -> str:
    s = (name_or_arch or "").lower()
    for family in _TEMPLATES:
        if family in s:
            return family
    return "default"


def get_prompt_template(model_name: str, architecture: str | None = None) -> dict:
    """Return {"user": "...{prompt}...", "stop": [...]} for the given model.

    Detection prefers the model's NAME over its architecture: Mistral models
    report arch="llama" in GGUF metadata (Mistral is built on the Llama
    architecture) but use [INST] templates, not Llama-3 chat tokens. The name
    is the more reliable signal for choosing a chat template."""
    family = _family_for(model_name)
    if family == "default":
        family = _family_for(architecture or "")
    return _TEMPLATES.get(family, _DEFAULT)


def format_prompt(model_name: str, architecture: str | None, user_prompt: str) -> tuple[str, list[str]]:
    tmpl = get_prompt_template(model_name, architecture)
    return tmpl["user"].format(prompt=user_prompt), list(tmpl["stop"])
