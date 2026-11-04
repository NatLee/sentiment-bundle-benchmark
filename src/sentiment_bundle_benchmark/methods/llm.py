"""Zero- or few-shot classification with a small instruction-tuned LLM, run locally.

The model never generates free text. We build a chat prompt that ends right where the answer
starts, run one forward pass, and compare the probability of each label word's first token
(lower-case and capitalised spellings both count). That is deterministic and cheap.
"""

from __future__ import annotations

import numpy as np
import torch

from ..config import MethodConfig
from ._torch import peak_mb, pick_device, pick_dtype, reset_peak, seed_everything, sync, weights_of
from .base import MethodInput, MethodOutput, Stopwatch
from .prompting import build_prompt, label_token_ids, pick_demos, truncate


def run(method: MethodConfig, data: MethodInput) -> MethodOutput:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    seed_everything(data.seed)
    device = pick_device()
    dtype = pick_dtype(device)
    name, revision = method.require("model"), method.get("revision")
    spec, train, test = data.spec, data.splits.train, data.splits.test
    batch_size = int(method.get("batch_size", 8))
    max_input = int(method.get("max_input_tokens", 512))

    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    tokenizer.padding_side = "left"  # so the last position of every row is the answer slot
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=dtype).to(device).eval()
    choices = label_token_ids(tokenizer, spec.labels)

    demos = [
        (truncate(tokenizer, text, int(method.get("max_demo_tokens", 96))), spec.labels[label])
        for text, label in pick_demos(train, int(method.get("shots", 0)), len(spec.labels), data.seed)
    ]

    def to_prompt(text: str) -> str:
        message = build_prompt(spec, truncate(tokenizer, text, max_input), demos)
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": message}], tokenize=False, add_generation_prompt=True,
            enable_thinking=False,  # Qwen3 templates: answer directly, no reasoning block
        )

    infer_clock = Stopwatch()
    reset_peak(device)
    with infer_clock.running(), torch.inference_mode():
        prompts = [tokenizer.encode(to_prompt(t), add_special_tokens=False) for t in test["text"]]
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        scores = np.empty((len(prompts), len(choices)), dtype=np.float32)
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            batch = tokenizer.pad({"input_ids": [prompts[i] for i in index]}, return_tensors="pt").to(device)
            positions = (batch["attention_mask"].cumsum(-1) - 1).clamp(min=0)
            logits = model(**batch, position_ids=positions, logits_to_keep=1).logits[:, -1, :].float()
            logprobs = torch.log_softmax(logits, dim=-1)
            for column, ids in enumerate(choices):
                scores[index, column] = torch.logsumexp(logprobs[:, ids], dim=-1).cpu().numpy()
        sync(device)

    params, size_mb = weights_of(model)
    lengths = [len(p) for p in prompts]
    return MethodOutput(
        predictions=scores.argmax(1).tolist(),
        train_seconds=0.0,
        infer_seconds=infer_clock.seconds,
        params=params,
        size_mb=size_mb,
        peak_vram_mb=peak_mb(device),
        device=str(device),
        details={
            "shots": len(demos),
            "prompt_tokens_mean": float(np.mean(lengths)),
            "prompt_tokens_max": int(max(lengths)),
            "example_prompt": to_prompt(test["text"].iloc[0]),
        },
    )
