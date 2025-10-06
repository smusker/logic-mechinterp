#!/usr/bin/env python3
"""
Reasoning Patch — Interventions-Only Script (no flags)

This script:
  1) Loads a large HF model + tokenizer (defaults to "openai/gpt-oss-20b").
  2) Builds a Pyvene Intervenable at a chosen layer (default: 12).
  3) Generates a small two-digit dataset.
  4) Runs the two-digit patching loop and writes "patch_double_digit_reasoning.csv".

Requirements:
  - torch
  - transformers
  - pyvene
  - A machine with enough RAM/disk for the selected model.
"""

from __future__ import annotations
import random
import csv
import json
import sys

# ---- Utilities ----
def set_seed(seed: int = 42) -> None:
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    random.seed(seed)

def _lazy_imports():
    mods = {}
    import importlib
    try:
        mods["torch"] = importlib.import_module("torch")
    except Exception as e:
        raise RuntimeError(f"Missing dependency: torch ({e})")
    try:
        tf = importlib.import_module("transformers")
        mods["AutoModelForCausalLM"] = getattr(tf, "AutoModelForCausalLM")
        mods["AutoTokenizer"] = getattr(tf, "AutoTokenizer")
        mods["Mxfp4Config"] = getattr(tf, "Mxfp4Config", None)
    except Exception as e:
        raise RuntimeError(f"Missing dependency: transformers ({e})")
    try:
        py = importlib.import_module("pyvene")
        mods["IntervenableModel"] = getattr(py, "IntervenableModel")
        mods["VanillaIntervention"] = getattr(py, "VanillaIntervention")
        mods["RepresentationConfig"] = getattr(py, "RepresentationConfig")
        mods["IntervenableConfig"] = getattr(py, "IntervenableConfig")
    except Exception as e:
        raise RuntimeError(f"Missing dependency: pyvene ({e})")
    return mods

# ---- Core helpers ----
def load_model_and_tokenizer(model_id: str = "openai/gpt-oss-20b", cache_dir: str | None = None):
    mods = _lazy_imports()
    torch = mods["torch"]
    AutoModelForCausalLM = mods["AutoModelForCausalLM"]
    AutoTokenizer = mods["AutoTokenizer"]
    Mxfp4Config = mods["Mxfp4Config"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    quant_cfg = None
    if Mxfp4Config is not None:
        try:
            quant_cfg = Mxfp4Config(dequantize=True)
        except Exception:
            quant_cfg = None

    model_kwargs = dict(
        attn_implementation="eager",
        dtype=getattr(torch, "bfloat16", None),
        quantization_config=quant_cfg,
        use_cache=False,
        device_map="cpu" if device == "cpu" else None,
        cache_dir=cache_dir,
    )
    print(f"[load] Model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, device

def build_intervenable(model, device: str, layer: int = 12):
    mods = _lazy_imports()
    IntervenableModel = mods["IntervenableModel"]
    RepresentationConfig = mods["RepresentationConfig"]
    IntervenableConfig = mods["IntervenableConfig"]
    VanillaIntervention = mods["VanillaIntervention"]

    config = IntervenableConfig(
        model_type=type(model),
        representations=[
            RepresentationConfig(layer, "block_output"),
        ],
        intervention_types=[VanillaIntervention],
    )
    intervenable = IntervenableModel(config, model)
    if hasattr(intervenable, "set_device"):
        intervenable.set_device(device)
    if hasattr(intervenable, "disable_model_gradients"):
        intervenable.disable_model_gradients()
    return intervenable

def build_two_digit_reasoning_template() -> str:
    return (
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\\n"
        "Knowledge cutoff: 2024-06\\n"
        "Current date: 2025-06-28\\n\\n"
        "Reasoning: high\\n\\n"
        "# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
        "<|start|>user<|message|>What is 1TEN1ONE+2TEN2ONE?<|end|>"
        "<|start|>assistant<|channel|>analysis<|message|>"
        "The user: \\\"What is 1TEN1ONE+2TEN2ONE?\\\" Simple addition. 1TEN0 + 2TEN0 ="
    )

def make_two_digit_dataset(n: int = 100, seed: int = 42):
    random.seed(seed)
    list1 = [(random.randint(1, 9), random.randint(0, 9), random.randint(1, 9), random.randint(0, 9)) for _ in range(n)]
    list2 = [(random.randint(1, 9), random.randint(0, 9), random.randint(1, 9), random.randint(0, 9)) for _ in range(n)]
    factual_sums = [[a1*10 + a2 + a3*10 + a4, b1*10 + b2 + b3*10 + b4] for (a1, a2, a3, a4), (b1, b2, b3, b4) in zip(list1, list2)]
    num_swap_sums = [[a1*10 + a2 + b3*10 + b4, b1*10 + b2 + a3*10 + a4] for (a1, a2, a3, a4), (b1, b2, b3, b4) in zip(list1, list2)]
    digit_swap_sums = [[a1*10 + b2 + a3*10 + b4, b1*10 + a2 + b3*10 + a4] for (a1, a2, a3, a4), (b1, b2, b3, b4) in zip(list1, list2)]
    ind_swap_sums = [[a1*10 + a2 + a3*10 + b4, a1*10 + a2 + b3*10 + a4, a1*10 + b2 + a3*10 + a4, b1*10 + a2 + a3*10 + a4]
                     for (a1, a2, a3, a4), (b1, b2, b3, b4) in zip(list1, list2)]
    return list1, list2, factual_sums, num_swap_sums, digit_swap_sums, ind_swap_sums

def run_two_digit_patching_csv(tokenizer, intervenable, layer: int,
                               list1, list2, factual_sums, num_swap_sums, digit_swap_sums, ind_swap_sums,
                               csv_path: str = "patch_double_digit_reasoning.csv",
                               max_examples: int = 10,
                               token_span = (84, 97)):
    mods = _lazy_imports()
    torch = mods["torch"]
    template = build_two_digit_reasoning_template()

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer","token_position","token","base_addition","source_addition","prediction","output","factual_sum","num_swap_sum","digit_swap_sum","ind_swap_sum"])

    for idx, (base_nums, source_nums, factual_sum, num_swap_sum, digit_swap_sum, ind_swap_sum) in enumerate(
        zip(list1, list2, factual_sums, num_swap_sums, digit_swap_sums, ind_swap_sums)
    ):
        if idx >= max_examples:
            break

        a1ten, a1one, a2ten, a2one = base_nums
        b1ten, b1one, b2ten, b2one = source_nums

        base = (template.replace("1TEN", f"{a1ten}").replace("1ONE", f"{a1one}").replace("2TEN", f"{a2ten}").replace("2ONE", f"{a2one}"))
        source = (template.replace("1TEN", f"{b1ten}").replace("1ONE", f"{b1one}").replace("2TEN", f"{b2ten}").replace("2ONE", f"{b2one}"))

        base_tokens = tokenizer(base, return_tensors="pt")
        source_tokens = tokenizer(source, return_tensors="pt")

        for tok_pos in range(token_span[0], token_span[1]):
            output_str = ""
            pred_str = ""
            base_tokens_copy = base_tokens["input_ids"].clone()
            source_tokens_copy = source_tokens["input_ids"].clone()

            token = tokenizer.decode(base_tokens_copy[0][tok_pos])

            while pred_str != "<|return|>":
                with torch.no_grad():
                    _, counterfactual_outputs = intervenable(
                        base={"input_ids": base_tokens_copy},
                        sources=[{"input_ids": source_tokens_copy}],
                        unit_locations={"sources->base": tok_pos},
                    )
                pred_tok = counterfactual_outputs.logits[0, -1].argmax(dim=-1)
                if pred_tok.item() == 200002:
                    break
                pred_str = tokenizer.decode(pred_tok)
                output_str += pred_str
                base_tokens_copy = torch.cat([base_tokens_copy, pred_tok.unsqueeze(0).unsqueeze(0)], dim=1)
                source_tokens_copy = torch.cat([source_tokens_copy, pred_tok.unsqueeze(0).unsqueeze(0)], dim=1)

            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    layer, tok_pos, token,
                    f"{a1ten}{a1one}+{a2ten}{a2one}", f"{b1ten}{b1one}+{b2ten}{b2one}",
                    pred_str, output_str.replace("\\n", "\\\\n"),
                    factual_sum, num_swap_sum, digit_swap_sum, ind_swap_sums
                ])

def main():
    set_seed(42)
    print("[start] Reasoning Patch — Interventions Only")

    # Load model + tokenizer
    model, tokenizer, device = load_model_and_tokenizer(model_id="openai/gpt-oss-20b", cache_dir=None)
    print(f"[model] Loaded. Device preference resolved to: {'cuda' if device=='cuda' else 'cpu'}")

    # Build pyvene Intervenable
    layer = 12
    intervenable = build_intervenable(model, device, layer=layer)
    print("[pyvene] Intervenable constructed.")

    # Dataset
    list1, list2, factual_sums, num_swap_sums, digit_swap_sums, ind_swap_sums = make_two_digit_dataset(n=100, seed=42)
    print(f"[data] Dataset ready: {len(list1)} pairs.")

    # Run patching
    out_csv = "patch_double_digit_reasoning.csv"
    run_two_digit_patching_csv(
        tokenizer, intervenable, layer,
        list1, list2, factual_sums, num_swap_sums, digit_swap_sums, ind_swap_sums,
        csv_path=out_csv, max_examples=10, token_span=(84, 97)
    )
    print(f"[done] Wrote CSV: {out_csv}")

if __name__ == "__main__":
    main()
