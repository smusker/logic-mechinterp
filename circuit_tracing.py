#!/usr/bin/env python3
"""
circuit_tracing.py
------------------
Trace (and visualise) which attention heads in Meta-Llama-3-8B are
causally important for
  1. the Boolean “True/False” token, and
  2. the first token after the literal phrase “if and only if”
in a synthetic rule-induction prompt.

• Works with the current TransformerLens releases (≥ 1.11).
• First tries Anthropic’s circuit-tracer; if unavailable, falls back
  to an internal head-ablation tracer (≈ 1½ min on A100 for LL-3-8B).
• Produces `.gpickle` graphs and nice `.png` spring-layouts in
  the output directory.

Usage
=====
$ pip install --upgrade torch transformer_lens networkx matplotlib
$ # optional but *much* faster if you can install it:
$ pip install circuit-tracer
$ python circuit_tracing.py --prompt-file prompt.txt --output-dir trace_outputs
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import networkx as nx
import torch
from transformer_lens import HookedTransformer

###############################################################################
# Optional fast path – Anthropic’s circuit-tracer (pip install circuit-tracer)
#pip install git+https://github.com/safety-research/circuit-tracer.git
###############################################################################
try:
    from circuit_tracer.trace_circuits import trace_circuits as ct_trace_circuits
    print("I ran") #looks like this isn't running
#from circuit_tracer import trace_circuits as ct_trace_circuits  # type: ignore
except ModuleNotFoundError:  # library not installed – we’ll fall back
    ct_trace_circuits = None

###############################################################################
# Logging helpers
###############################################################################


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


###############################################################################
# Tiny utilities
###############################################################################


def load_prompt(path: str | Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def find_subseq(full: List[int], sub: List[int]) -> Optional[int]:
    """Return first index i such that full[i:i+len(sub)] == sub, else None."""
    n, m = len(full), len(sub)
    for i in range(n - m + 1):
        if full[i : i + m] == sub:
            return i
    return None


###############################################################################
# Slow but self-contained head-ablation circuit tracer
###############################################################################


def _fallback_trace_circuits(
    model: HookedTransformer,
    tokens: torch.Tensor,
    dest_pos: int,
    label_id: int,
    max_edges: int = 500,
    device: str = "cuda",
) -> nx.DiGraph:
    """
    Very simple circuit tracer that assigns an *importance weight* to every
    (layer, head) by measuring how much zero-ablation of that head’s value
    stream decreases the logit of `label_id` at `dest_pos`.
    """
    model.eval()
    tokens = tokens.to(device)

    # Baseline logits at destination position
    with torch.no_grad():
        baseline_logits = model(tokens)[0, dest_pos]
    baseline_score = baseline_logits[label_id].item()

    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    importance = torch.zeros(n_layers, n_heads, device=device)

    # Iterate over heads (***slow but dependency-free***)
    for layer in range(n_layers):
        for head in range(n_heads):
            hook_name = f"blocks.{layer}.attn.hook_z"

            def ablate_value(z, hook, h=head):
                z = z.clone()
                z[:, :, h, :] = 0.0
                return z

            with torch.no_grad():
                out_logits = model.run_with_hooks(
                    tokens, fwd_hooks=[(hook_name, ablate_value)]
                )[0][0, dest_pos]

            delta = baseline_score - out_logits[label_id].item()
            importance[layer, head] = max(delta, 0.0)  # keep non-negative gains only

    # Build a star-shaped attribution graph
    G = nx.DiGraph()
    dest_node = ("logit", dest_pos)
    G.add_node(dest_node, weight=baseline_score)

    flat: List[Tuple[int, int, float]] = [
        (l, h, importance[l, h].item())
        for l in range(n_layers)
        for h in range(n_heads)
        if importance[l, h] > 0
    ]
    flat.sort(key=lambda t: -t[2])
    for l, h, w in flat[:max_edges]:
        node = (l, h)  # tuple => nicely filtered later for highlighting
        G.add_node(node, weight=w)
        G.add_edge(node, dest_node, weight=w)
    return G


###############################################################################
# Main
###############################################################################


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Trace important heads in Llama-3-8B for a rule-induction prompt."
    )
    parser.add_argument("--prompt-file", default="prompt.txt")
    parser.add_argument("--output-dir", default="trace_outputs")
    parser.add_argument("--max-edges", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading prompt …")
    prompt_text: str = load_prompt(args.prompt_file)

    ##################################################
    # 1) Load the model
    ##################################################
    logging.info("Loading Meta-Llama-3-8B via TransformerLens …")
    model = HookedTransformer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        device=args.device,
        fold_ln=True,
        center_writing_weights=True,
        trust_remote_code=True,
    ).eval()

    ##################################################
    # 2) Ask for the Boolean label prediction
    ##################################################
    tok_prompt = model.to_tokens(prompt_text, prepend_bos=False)
    with torch.no_grad():
        logits_prompt = model(tok_prompt)[0]
    next_logits = logits_prompt[0, -1]
    pred_id: int = int(next_logits.argmax().item())
    pred_str: str = model.to_string(torch.tensor([[pred_id]], device=args.device)).strip()
    logging.info("Model predicts Boolean label: %s", pred_str)

    ##################################################
    # 3) Force a continuation up to “…if and only if ”
    ##################################################
    forced_suffix = (
        f" -> {pred_str}\n"
        "Based on your examples, I've learned that an object should be labeled "
        "'True' if and only if "
    )
    full_text = prompt_text + forced_suffix
    tok_full = model.to_tokens(full_text, prepend_bos=False)
    seq: List[int] = tok_full[0].tolist()

    # Destination A: Boolean token position
    bool_token_id = model.tokenizer(f" {pred_str}", add_special_tokens=False).input_ids[0]
    dest_bool = seq.index(bool_token_id)

    # Destination B: first token after “if and only if”
    phrase_ids = model.tokenizer("if and only if", add_special_tokens=False).input_ids
    start = find_subseq(seq, phrase_ids)
    if start is None:
        raise ValueError("'if and only if' not found in sequence!")
    dest_var1 = start + len(phrase_ids)

    ##################################################
    # 4) Run once to cache activations (needed by circuit-tracer)
    ##################################################
    with torch.no_grad():
        logits_full, cache = model.run_with_cache(tok_full)

    ##################################################
    # 5) Choose tracer implementation
    ##################################################
    def run_tracer(
        dest_pos: int, label_token_id: int, tag: str
    ) -> nx.DiGraph:  # small closure
        if ct_trace_circuits is not None:
            logging.info("Using fast circuit-tracer backend for %s …", tag)
            return ct_trace_circuits(
                model=model,
                cache=cache,
                src_pos=list(range(dest_pos)),
                dest_pos=dest_pos,
                max_edges=args.max_edges,
            )
        else:
            logging.info("Using built-in (slow) head-ablation tracer for %s …", tag)
            return _fallback_trace_circuits(
                model,
                tok_full,
                dest_pos=dest_pos,
                label_id=label_token_id,
                max_edges=args.max_edges,
                device=args.device,
            )

    ##################################################
    # 6) Trace both destinations
    ##################################################
    for name, dest_pos, token_id in [
        ("Bool", dest_bool, bool_token_id),
        ("Var1", dest_var1, int(logits_full[0, dest_var1].argmax().item())),
    ]:
        G = run_tracer(dest_pos, token_id, name)

        # 6a) Save raw graph
        gpath = out_dir / f"trace_{name}.gpickle"
        nx.write_gpickle(G, gpath)
        logging.info("Saved NetworkX graph ➜ %s", gpath)

        # 6b) Quick spring layout PNG
        logging.info("Rendering PNG for %s …", name)
        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(G, seed=args.seed)
        nx.draw_networkx_edges(G, pos, alpha=0.15)

        weights = nx.get_node_attributes(G, "weight")
        head_nodes = [(n, w) for n, w in weights.items() if isinstance(n, tuple)]
        head_nodes.sort(key=lambda p: -p[1])
        top_nodes = [n for n, _ in head_nodes[:20]]
        sizes = [weights[n] * 200 for n in top_nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=top_nodes, node_size=sizes)
        nx.draw_networkx_labels(
            G, pos, labels={n: f"{n}" for n in top_nodes}, font_size=7
        )
        plt.axis("off")
        png_path = out_dir / f"trace_{name}.png"
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        logging.info("Saved PNG ➜ %s", png_path)

    logging.info("🎉  All done!")


if __name__ == "__main__":
    main()
