#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Produce a CSV "table of outputs under interventions" by reproducing the
position-aligned intervention sweep from the notebook.

It:
  - builds a small synthetic two-digit addition dataset
  - formats prompts using the same ChatML-ish template seen in the notebook
  - wraps `openai/gpt-oss-20b` with Pyvene's IntervenableModel
  - for each example and token position (84..96 by default), performs a single
    sources->base intervention and greedily decodes until <|return|>
  - writes one CSV row per intervention with rich metadata

Tested with:
  - Python 3.10+
  - torch 2.4+
  - transformers 4.44+
  - pyvene 0.1.x

Install (example):
  pip install torch transformers pyvene datasets tqdm

Run:
  python intervention_table.py --out patch_double_digit_reasoning.csv --pairs 10

Tip: on an A100, keep `--device cuda` (default if CUDA is available).
"""

import argparse
import csv
import json
import os
import random
import sys
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional: MXFP4 quantization is nice-to-have; we fall back gracefully if missing.
try:
    from transformers import Mxfp4Config  # type: ignore
    HAS_MXFP4 = True
except Exception:
    HAS_MXFP4 = False

# ---- Prompt templates (match the notebook) ----

TWO_DIGIT_REASONING_TEMPLATE = (
    "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
    "Knowledge cutoff: 2024-06\n"
    "Current date: 2025-06-28\n"
    "\n"
    "Reasoning: high\n"
    "\n"
    "# Valid channels: analysis, commentary, final. Channel must be included for every message."
    "<|end|><|start|>user<|message|>What is 1TEN1ONE+2TEN2ONE?<|end|><|start|>assistant<|channel|>analysis<|message|>"
    "The user: \"What is 1TEN1ONE+2TEN2ONE?\" Simple addition. 1TEN0 + 2TEN0 ="
)

# These are only used to match stopping conditions in decoding
STOP_STR = "<|return|>"


# ---- Data generation identical in spirit to your notebook ----

def make_pairs(n: int, seed: int = 42) -> List[Tuple[Tuple[int,int,int,int], Tuple[int,int,int,int]]]:
    rng = random.Random(seed)
    list1 = [(rng.randint(1, 9), rng.randint(0, 9), rng.randint(1, 9), rng.randint(0, 9)) for _ in range(n)]
    list2 = [(rng.randint(1, 9), rng.randint(0, 9), rng.randint(1, 9), rng.randint(0, 9)) for _ in range(n)]
    return list1, list2


def sums_from_tuple(t):
    a1, a2, a3, a4 = t
    # The notebook's "sum" strings actually represent concatenated two-digit numbers added:
    # (a1 a2) + (a3 a4)
    left = a1 * 10 + a2
    right = a3 * 10 + a4
    return left + right


def build_derived_sums(list1, list2):
    factual = []
    num_swap = []
    digit_swap = []
    ind_swap = []
    for (a1, a2, a3, a4), (b1, b2, b3, b4) in zip(list1, list2):
        factual.append([a1*10 + a2 + a3*10 + a4, b1*10 + b2 + b3*10 + b4])
        num_swap.append([a1*10 + a2 + b3*10 + b4, b1*10 + b2 + a3*10 + a4])
        digit_swap.append([a1*10 + b2 + a3*10 + b4, b1*10 + a2 + b3*10 + a4])
        ind_swap.append([
            a1*10 + a2 + a3*10 + b4,
            a1*10 + a2 + b3*10 + a4,
            a1*10 + b2 + a3*10 + a4,
            b1*10 + a2 + a3*10 + a4
        ])
    return factual, num_swap, digit_swap, ind_swap


# ---- Pyvene wrapper ----

def make_intervenable(model, layer: int, device: str):
    from pyvene import IntervenableModel, VanillaIntervention, RepresentationConfig, IntervenableConfig
    config = IntervenableConfig(
        model_type=type(model),
        representations=[
            RepresentationConfig(layer, "block_output"),
        ],
        intervention_types=[VanillaIntervention],
    )
    intervenable = IntervenableModel(config, model)
    intervenable.set_device(device)
    intervenable.disable_model_gradients()
    return intervenable


# ---- Decoding loop with intervention ----

@torch.no_grad()
def run_intervention_sample(
    intervenable,
    tokenizer,
    base_ids: torch.Tensor,
    src_ids: torch.Tensor,
    tok_pos: int,
    max_new_tokens: int = 64,
):
    """
    Perform one intervention (sources->base at tok_pos), greedy decode to STOP_STR/EOS/max length.
    Returns (last_pred_str, full_output_str).
    """
    # We expand both base and source with generated tokens step-by-step (as in the notebook).
    base_cur = base_ids.clone()
    src_cur = src_ids.clone()

    out_str = ""
    last_pred_str = ""
    seen_stop = False

    eos_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        _, counterfactual_outputs = intervenable(
            base={"input_ids": base_cur},
            sources=[{"input_ids": src_cur}],
            unit_locations={"sources->base": tok_pos},
        )
        pred_tok = counterfactual_outputs.logits[0, -1].argmax(dim=-1)

        if eos_id is not None and pred_tok.item() == eos_id:
            # Some tokenizers use EOS instead of "<|return|>"
            last_pred_str = "<|eos|>"
            break

        step_str = tokenizer.decode(pred_tok)
        out_str += step_str
        last_pred_str = step_str

        if STOP_STR in step_str:
            seen_stop = True
            break

        # Append token to both base and source sequences (mirroring the notebook)
        pred_tok_batched = pred_tok.unsqueeze(0).unsqueeze(0)  # [1,1]
        base_cur = torch.cat([base_cur, pred_tok_batched], dim=1)
        src_cur  = torch.cat([src_cur,  pred_tok_batched], dim=1)

    if not seen_stop and last_pred_str == "":
        last_pred_str = "<|none|>"

    return last_pred_str, out_str.replace("\n", "\\n")


# ---- Utilities ----

def fmt_prompt(tpl: str, a1, a2, a3, a4) -> str:
    # Replace placeholders to match the notebook’s exact format
    # NOTE: The template uses 1TEN / 1ONE / 2TEN / 2ONE and also includes "1TEN0 + 2TEN0"
    s = tpl.replace("1TEN", f"{a1}").replace("1ONE", f"{a2}").replace("2TEN", f"{a3}").replace("2ONE", f"{a4}")
    return s


def main():
    parser = argparse.ArgumentParser(description="Recreate the table of outputs under interventions.")
    parser.add_argument("--model", default="openai/gpt-oss-20b", help="HF model id")
    parser.add_argument("--cache", default=None, help="HF cache dir (optional)")
    parser.add_argument("--out", default="patch_double_digit_reasoning.csv", help="Output CSV path")
    parser.add_argument("--pairs", type=int, default=10, help="How many base/source pairs to run")
    parser.add_argument("--layer", type=int, default=12, help="Layer index to intervene on")
    parser.add_argument("--tok_start", type=int, default=84, help="First token position to intervene")
    parser.add_argument("--tok_end_inclusive", type=int, default=96, help="Last token position (inclusive)")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Decode cap per intervention")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"], help="Force device (default: auto)")
    parser.add_argument("--print_preview", action="store_true", help="Pretty-print first example table to stdout")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    # Device selection
    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model load kwargs
    model_kwargs = dict(
        attn_implementation="eager",
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        use_cache=False,
        cache_dir=args.cache,
        device_map="auto" if device == "cuda" else "cpu",
    )
    if HAS_MXFP4:
        try:
            model_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
        except Exception:
            pass

    print(f"[INFO] Loading model: {args.model} on {device} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    # Create intervenable wrapper
    intervenable = make_intervenable(model, layer=args.layer, device=device)

    # Build data
    list1, list2 = make_pairs(args.pairs, seed=args.seed)
    factual_sums, num_swap_sums, digit_swap_sums, ind_swap_sums = build_derived_sums(list1, list2)

    # Prepare CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    csv_f = open(args.out, "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow([
        "layer",
        "token_position",
        "token",
        "base_addition",
        "source_addition",
        "prediction",
        "output",
        "factual_sum",
        "num_swap_sum",
        "digit_swap_sum",
        "ind_swap_sum",
    ])

    # Optional: keep a small preview log for stdout
    preview_rows = []

    # Main loop
    TOK_START = args.tok_start
    TOK_END = args.tok_end_inclusive

    for idx, (base_nums, src_nums) in enumerate(zip(list1, list2)):
        a1ten, a1one, a2ten, a2one = base_nums
        b1ten, b1one, b2ten, b2one = src_nums

        base_prompt = fmt_prompt(TWO_DIGIT_REASONING_TEMPLATE, a1ten, a1one, a2ten, a2one)
        src_prompt  = fmt_prompt(TWO_DIGIT_REASONING_TEMPLATE, b1ten, b1one, b2ten, b2one)

        base_toks = tokenizer(base_prompt, return_tensors="pt").to(device)
        src_toks  = tokenizer(src_prompt,  return_tensors="pt").to(device)

        base_ids = base_toks["input_ids"]
        src_ids  = src_toks["input_ids"]

        # For replicability with your notebook, we print the labels of the window (optional)
        # labels_slice = tokenizer.convert_ids_to_tokens(base_ids[0][TOK_START:TOK_END+1])
        # print(labels_slice)

        for tok_pos in range(TOK_START, TOK_END + 1):
            token_str = tokenizer.decode(base_ids[0][tok_pos])

            last_pred, full_out = run_intervention_sample(
                intervenable=intervenable,
                tokenizer=tokenizer,
                base_ids=base_ids,
                src_ids=src_ids,
                tok_pos=tok_pos,
                max_new_tokens=args.max_new_tokens,
            )

            # Pack metadata like the notebook
            base_add = f"{a1ten}{a1one}+{a2ten}{a2one}"
            src_add  = f"{b1ten}{b1one}+{b2ten}{b2one}"
            factual  = factual_sums[idx]
            num_swap = num_swap_sums[idx]
            digit_sw = digit_swap_sums[idx]
            ind_sw   = ind_swap_sums[idx]

            writer.writerow([
                args.layer,
                tok_pos,
                token_str,
                base_add,
                src_add,
                last_pred,
                full_out,
                factual,
                num_swap,
                digit_sw,
                ind_sw
            ])

            # Save first example rows for console preview if requested
            if args.print_preview and idx == 0:
                preview_rows.append({
                    "pos": tok_pos,
                    "token": token_str,
                    "prediction": last_pred,
                    "output": full_out,
                })

    csv_f.close()
    print(f"[DONE] Wrote CSV: {os.path.abspath(args.out)}")

    if args.print_preview:
        # Pretty-print a compact table for the first example
        try:
            from tabulate import tabulate  # optional
            table = [(r["pos"], r["token"], r["prediction"], r["output"]) for r in preview_rows]
            print("\nFirst example — per-position interventions")
            print(tabulate(table, headers=["pos", "token", "prediction", "output"], tablefmt="github"))
        except Exception:
            # Fallback printing
            print("\nFirst example — per-position interventions")
            print("pos | token | prediction | output")
            print("-" * 60)
            for r in preview_rows:
                print(f"{r['pos']} | {repr(r['token'])} | {repr(r['prediction'])} | {repr(r['output'][:80])}...")


if __name__ == "__main__":
    main()
