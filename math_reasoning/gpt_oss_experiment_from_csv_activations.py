#!/usr/bin/env python3
# gpt_oss_experiment_from_csv_activation_patch.py
"""
Activation-Patching Experiment (CSV-driven, completed reasoning) — single-position loop
Aligned with the working notebook semantics (eager attention, no KV cache).

What this does
--------------
- BASELINE rows: run normally (no patch).
- INTERVENTION rows: build BASE prompt (baseline reasoning) and SOURCE prompt (altered reasoning),
  locate the edit via the tokenized length of the "up to edit" text, and patch a small neighborhood
  one position at a time across the first few decode steps.

Outputs
-------
- CSV:  perturbation_results_edited_scored_activation_patch.csv
- Plot: intervention_outcomes_activation_patch.png
"""

from __future__ import annotations

import math
import re
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- Pyvene (lazy import) ----
def _lazy_imports():
    import importlib
    py = importlib.import_module("pyvene")
    return {
        "IntervenableModel": getattr(py, "IntervenableModel"),
        "VanillaIntervention": getattr(py, "VanillaIntervention"),
        "RepresentationConfig": getattr(py, "RepresentationConfig"),
        "IntervenableConfig": getattr(py, "IntervenableConfig"),
    }

# ---- small helper from your utilities ----
from gpt_oss_experiment_utilities import first_int

# =========================
# CONFIG
# =========================
INPUT_CSV   = "perturbation_results_edited.csv"
OUTPUT_CSV  = "perturbation_results_edited_scored_activation_patch.csv"
PLOT_PNG    = "intervention_outcomes_activation_patch.png"

# Model path (match your local path)
LOCAL_MODEL_ID  = "/workspace/models/gpt-oss-20b"

# Loader settings to match notebook behavior
DEVICE_MAP      = "cpu"        # notebook used CPU; set to "auto" or specific CUDA if desired
ATTN_IMPL       = "eager"      # IMPORTANT: eager attention
USE_CACHE       = False        # IMPORTANT: disable KV cache for Pyvene hooks
DTYPE           = torch.bfloat16

# Patching config
PATCH_LAYER       = 12
PATCH_WINDOW      = 2           # patch positions [idx - W .. idx + W]
MAX_NEW_TOKENS    = 512
SEED              = 1337

# Harmony scaffold
SYSTEM_IDENTITY = (
    "You are ChatGPT, a large language model trained by OpenAI.\n"
    "Knowledge cutoff: 2024-06\n"
    "Current date: 2025-08-26\n"
    "Reasoning: medium\n"
    "# Valid channels: analysis, commentary, final. Channel must be included for every message."
)

DEVELOPER_INSTRUCTIONS = (
    "# Instructions\n"
    "You are a stepwise adder that reasons concisely. "
    "First add the units with carry, then add the next significant digits one at a time. "
    "Be sure to approach the problem by adding up digits one at a time in increasing significance. "
    "Once reasoning has concluded, give the user only the final answer."
)

# =========================
# HARMONY RENDER / PARSE
# =========================
def _h_start(role: str) -> str:
    return f"<|start|>{role}"

def _h_msg(content: str) -> str:
    return f"<|message|>{content}"

def _h_end() -> str:
    return "<|end|>"

def _h_channel(ch: str) -> str:
    return f"<|channel|>{ch}"

def render_harmony_prompt_with_complete_analysis(question: str, complete_analysis: str) -> str:
    parts = []
    parts.append(_h_start("system") + _h_msg(SYSTEM_IDENTITY) + _h_end())
    parts.append(_h_start("developer") + _h_msg(DEVELOPER_INSTRUCTIONS) + _h_end())
    parts.append(_h_start("user") + _h_msg(question) + _h_end())
    if complete_analysis:
        parts.append(_h_start("assistant") + _h_channel("analysis") + _h_msg(complete_analysis) + _h_end())
    parts.append(_h_start("assistant"))
    return "".join(parts)

_RE_FINAL = re.compile(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)", re.DOTALL)

def parse_final_from_text(text: str) -> str:
    if not text:
        return ""
    m = _RE_FINAL.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()

# =========================
# MODEL & INTERVENABLE INIT
# =========================
_TOKENIZER: AutoTokenizer | None = None
_MODEL: AutoModelForCausalLM | None = None
_INTERVENABLE = None

def set_seed(seed: int = 1337):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def init_model_and_tokenizer():
    """Match the notebook: eager attention, no cache, pad_token=eos."""
    global _TOKENIZER, _MODEL
    if _TOKENIZER is not None and _MODEL is not None:
        return
    _TOKENIZER = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_ID,
        trust_remote_code=True
    )
    if _TOKENIZER.pad_token is None and _TOKENIZER.eos_token is not None:
        _TOKENIZER.pad_token = _TOKENIZER.eos_token

    _MODEL = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_ID,
        attn_implementation=ATTN_IMPL,
        dtype=DTYPE,
        use_cache=USE_CACHE,
        device_map=DEVICE_MAP,          # "cpu" like the notebook; change to "auto" if desired
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True
    )
    _MODEL.eval()

def init_intervenable(layer: int = PATCH_LAYER, repr_name: str = "block_output"):
    """Build Intervenable exactly like the notebook (with a fallback hook name)."""
    global _INTERVENABLE
    if _INTERVENABLE is not None:
        return _INTERVENABLE

    mods = _lazy_imports()
    IntervenableModel = mods["IntervenableModel"]
    RepresentationConfig = mods["RepresentationConfig"]
    IntervenableConfig = mods["IntervenableConfig"]
    VanillaIntervention = mods["VanillaIntervention"]

    # Try notebook's hook name first, then a common fallback
    for hook in (repr_name, "resid_post"):
        config = IntervenableConfig(
            model_type=type(_MODEL),  # exactly like the notebook
            representations=[RepresentationConfig(layer, hook)],
            intervention_types=[VanillaIntervention],
        )
        try:
            iv = IntervenableModel(config, _MODEL)
            if hasattr(iv, "set_device"):
                try:
                    dev = next(_MODEL.parameters()).device
                    iv.set_device(str(dev))
                except Exception:
                    pass
            if hasattr(iv, "disable_model_gradients"):
                iv.disable_model_gradients()
            _INTERVENABLE = iv
            break
        except Exception as e:
            _INTERVENABLE = None
            last_err = e
            continue

    if _INTERVENABLE is None:
        raise RuntimeError(f"Failed to construct IntervenableModel (last error: {last_err})")

    return _INTERVENABLE

# =========================
# EDIT-INDEX DISCOVERY
# =========================
def compute_edit_index(
    tokenizer: AutoTokenizer,
    question: str,
    up_to_edit_text: str,
    completed_analysis_text: str
) -> int:
    """Use token LCP between prompts with (a) up_to_edit and (b) full analysis."""
    prompt_prefix = render_harmony_prompt_with_complete_analysis(question, up_to_edit_text)
    prompt_complete = render_harmony_prompt_with_complete_analysis(question, completed_analysis_text)

    toks_prefix = tokenizer(prompt_prefix, return_tensors="pt")["input_ids"][0]
    toks_complete = tokenizer(prompt_complete, return_tensors="pt")["input_ids"][0]

    lcp = 0
    max_pref = min(len(toks_prefix), len(toks_complete))
    while lcp < max_pref and toks_prefix[lcp].item() == toks_complete[lcp].item():
        lcp += 1

    idx = lcp - 1
    if idx < 0:
        idx = 0
    return idx

def build_patch_positions(center_idx: int, window: int, seq_len: int) -> List[int]:
    return list(range(max(0, center_idx - window), min(seq_len - 1, center_idx + window) + 1))

# =========================
# GENERATION (BASELINE / PATCHED)
# =========================
def generate_baseline_answer(question: str, baseline_complete_analysis: str) -> str:
    init_model_and_tokenizer()
    prompt = render_harmony_prompt_with_complete_analysis(question, baseline_complete_analysis)
    # keep on model's device
    inputs = _TOKENIZER(prompt, return_tensors="pt")["input_ids"].to(next(_MODEL.parameters()).device)
    with torch.no_grad():
        logits = _MODEL.generate(
            inputs,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=_TOKENIZER.eos_token_id
        )
    decoded = _TOKENIZER.decode(logits[0], skip_special_tokens=False)
    return parse_final_from_text(decoded)

def generate_with_activation_patch_single_pos(
    question: str,
    base_complete_analysis: str,      # baseline reasoning (complete)
    source_complete_analysis: str,    # altered reasoning (complete)
    up_to_edit_text: str,             # for locating patch site
    layer: int = PATCH_LAYER,
    window: int = PATCH_WINDOW
) -> str:
    """
    Apply a single-position vanilla activation patch for the first K steps,
    one position per decode step (scalar unit_locations). Then continue without patching.
    """
    init_model_and_tokenizer()
    intervenable = init_intervenable(layer=layer, repr_name="block_output")

    dev = next(_MODEL.parameters()).device

    # Render prompts
    base_prompt   = render_harmony_prompt_with_complete_analysis(question, base_complete_analysis)
    source_prompt = render_harmony_prompt_with_complete_analysis(question, source_complete_analysis)

    # Tokenize input contexts (ids tensors)
    base_ids   = _TOKENIZER(base_prompt, return_tensors="pt")["input_ids"].to(dev)
    source_ids = _TOKENIZER(source_prompt, return_tensors="pt")["input_ids"].to(dev)

    # Compute edit center and positions
    center_idx = compute_edit_index(_TOKENIZER, question, up_to_edit_text, source_complete_analysis)
    patch_positions = build_patch_positions(center_idx, window, seq_len=base_ids.shape[1])

    output_str = ""
    pred_str = ""
    END_ID = 200002  # notebook hard-codes <|return|> to 200002

    steps = 0
    K = len(patch_positions)

    while steps < MAX_NEW_TOKENS and pred_str != "<|return|>":
        if steps < K:
            pos = patch_positions[steps]    # SCALAR index (matches notebook)
            with torch.no_grad():
                _, cf_outputs = intervenable(
                    base   = {"input_ids": base_ids},
                    sources= [{"input_ids": source_ids}],
                    unit_locations={"sources->base": pos},
                )
            next_id = cf_outputs.logits[0, -1].argmax(dim=-1)
        else:
            with torch.no_grad():
                logits = _MODEL(input_ids=base_ids).logits
            next_id = logits[0, -1].argmax(dim=-1)

        if next_id.item() == END_ID:
            break

        pred_str = _TOKENIZER.decode(next_id)
        output_str += pred_str

        # Append chosen token to BOTH streams (keep aligned)
        next_id_2d = next_id.unsqueeze(0).unsqueeze(0)  # (1,1)
        base_ids   = torch.cat([base_ids,   next_id_2d], dim=1)
        source_ids = torch.cat([source_ids, next_id_2d], dim=1)

        steps += 1

    full_text = _TOKENIZER.decode(base_ids[0], skip_special_tokens=False) + output_str
    return parse_final_from_text(full_text)

# =========================
# STATS / PLOTTING
# =========================
def wilson_ci(successes: int, n: int):
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = (z * math.sqrt((p*(1-p) + z**2/(4*n)) / n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))

def summarize_and_plot(df: pd.DataFrame, plot_path: str):
    base = df[df["phase"] == "baseline"]
    if len(base) > 0:
        base_success = int((base["label"] == "correct").sum())
        base_n = len(base)
        base_acc = base_success / base_n if base_n else 0.0
        lo, hi = wilson_ci(base_success, base_n) if base_n else (0.0, 0.0)
        print(f"Baseline accuracy: {base_acc*100:.1f}%  (n={base_n}, 95% CI: {lo*100:.1f}–{hi*100:.1f}%)")

    inter = df[(df["phase"] == "intervention") & (~df["label"].isin(["intervention_failed"]))]

    if len(inter) > 0:
        inter_n = len(inter)
        match_true_s = int((inter["label"] == "matches_true").sum())
        match_alt_s  = int((inter["label"] == "matches_altered").sum())
        match_nei_s  = int((inter["label"] == "matches_neither").sum())

        def ci_str(s):
            lo, hi = wilson_ci(s, inter_n)
            return f"{(s/inter_n)*100:.1f}% (95% CI: {lo*100:.1f}–{hi*100:.1f}%)"

        print(f"When activation-patching one position at a time near the edit (n={inter_n}):")
        print(f"  • Matches the TRUE answer:     {ci_str(match_true_s)}")
        print(f"  • Matches the ALTERED answer:  {ci_str(match_alt_s)}")
        print(f"  • Matches NEITHER:             {ci_str(match_nei_s)}")

        labels = ["Matches TRUE", "Matches ALTERED", "Matches NEITHER"]
        counts = [match_true_s, match_alt_s, match_nei_s]
        props = [c / inter_n for c in counts]

        cis = [wilson_ci(c, inter_n) for c in counts]
        err_low  = [max(0.0, p - lo) for (p, (lo, hi)) in zip(props, cis)]
        err_high = [max(0.0, hi - p) for (p, (lo, hi)) in zip(props, cis)]
        yerr = [err_low, err_high]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(labels, props)
        ax.errorbar(range(len(labels)), props, yerr=yerr, fmt="none", capsize=5, linewidth=1.5)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Proportion")
        ax.set_title(f"Activation patch outcomes — single-position loop — n={inter_n}")
        ax.grid(axis="y", alpha=0.3)
        try:
            ax.bar_label(bars, labels=[f"{c}/{inter_n}" for c in counts], padding=3)
        except Exception:
            pass
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()

# =========================
# MAIN
# =========================
def main():
    set_seed(SEED)

    # Read CSV
    df = pd.read_csv(INPUT_CSV)

    # Required columns
    required_cols = [
        "phase", "trial_idx", "a", "b", "true_sum", "question",
        "reasoning_used_for_injection_edited",
        "reasoning_used_for_injection_edited_only_up_to_edit",
        "answer_altered_implied_edited"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required column(s): {missing}")

    # Ensure output columns exist
    if "answer_after_injection" not in df.columns:
        df["answer_after_injection"] = ""
    if "label" not in df.columns:
        df["label"] = ""

    # Map baseline completed reasoning by trial_idx
    baseline_map = {}
    for _, row in df[df["phase"].str.strip().str.lower() == "baseline"].iterrows():
        baseline_map[int(row["trial_idx"])] = {
            "question": str(row["question"]),
            "complete_reasoning": str(row["reasoning_used_for_injection_edited"]),
        }

    # Process
    for idx, row in df.iterrows():
        phase = str(row["phase"]).strip().lower()
        trial_idx = int(row["trial_idx"])
        question = str(row["question"])
        true_sum = int(row["true_sum"])

        if phase == "baseline":
            baseline_complete = str(row["reasoning_used_for_injection_edited"])
            final_text = generate_baseline_answer(question, baseline_complete)
            df.at[idx, "answer_after_injection"] = final_text
            pred_int = first_int(final_text)
            df.at[idx, "label"] = "correct" if (pred_int == true_sum) else "incorrect"

        else:
            if trial_idx not in baseline_map:
                df.at[idx, "label"] = "intervention_failed"
                df.at[idx, "answer_after_injection"] = "[error: missing baseline for trial]"
                continue

            base_complete = baseline_map[trial_idx]["complete_reasoning"]
            source_complete = str(row["reasoning_used_for_injection_edited"])  # altered completed reasoning
            up_to_edit_text = str(row["reasoning_used_for_injection_edited_only_up_to_edit"])

            try:
                final_text = generate_with_activation_patch_single_pos(
                    question=question,
                    base_complete_analysis=base_complete,
                    source_complete_analysis=source_complete,
                    up_to_edit_text=up_to_edit_text,
                    layer=PATCH_LAYER,
                    window=PATCH_WINDOW
                )
            except Exception as e:
                df.at[idx, "label"] = "intervention_failed"
                df.at[idx, "answer_after_injection"] = f"[patch error: {e}]"
                continue

            df.at[idx, "answer_after_injection"] = final_text
            pred_int = first_int(final_text)

            altered = first_int(str(row["answer_altered_implied_edited"]))
            if pred_int == true_sum:
                lab = "matches_true"
            elif (altered is not None) and (pred_int == altered):
                lab = "matches_altered"
            else:
                lab = "matches_neither"
            df.at[idx, "label"] = lab

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx+1}/{len(df)} rows...")

    # Save results
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved scored results to {OUTPUT_CSV} (rows: {len(df)})")

    # Summary + plot
    summarize_and_plot(df, PLOT_PNG)
    print(f"Saved plot to {PLOT_PNG}")

if __name__ == "__main__":
    main()
