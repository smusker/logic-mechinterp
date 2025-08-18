# austin_circuit_trace.py
# Run this as a normal Python script (not a notebook).
# It will save HTML visualizations to disk (see prints at the end).
# This pulls the pre-constructed graph from neuronpedia 
# Adapted from https://github.com/safety-research/circuit-tracer/blob/main/demos/circuit_tracing_tutorial.ipynb / https://colab.research.google.com/github/safety-research/circuit-tracer/blob/main/demos/circuit_tracing_tutorial.ipynb#scrollTo=NKXAUFaF_ekG

from __future__ import annotations

import os
import sys
from pathlib import Path
from collections import namedtuple
from typing import List, Dict, Optional

import torch

# --- Paths: we expect the circuit-tracer repo to be cloned into repository/circuit-tracer ---
THIS_DIR = Path(__file__).resolve().parent
CT_ROOT = (THIS_DIR / "repository" / "circuit-tracer").resolve()

if not CT_ROOT.exists():
    raise SystemExit(
        f"Could not find circuit-tracer at {CT_ROOT}.\n"
        "Clone the repo into repository/circuit-tracer (see setup commands below)."
    )

# Ensure we can import the package and its demo utils
sys.path.insert(0, str(CT_ROOT))
sys.path.insert(0, str(CT_ROOT / "demos"))

from circuit_tracer import ReplacementModel
from utils import extract_supernode_features
from graph_visualization import create_graph_visualization, Supernode, InterventionGraph, Feature

# ---------------------------
# Helper: save HTML objects
# ---------------------------
def save_html(obj, path: Path) -> None:
    """Save an IPython.display.HTML object or raw HTML string to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    html_text = getattr(obj, "data", None)
    if html_text is None:  # maybe it's already a string
        html_text = str(obj)
    path.write_text(html_text, encoding="utf-8")
    print(f"Saved: {path}")

# ---------------------------
# Model load with dtype fallback
# ---------------------------
def pick_dtype() -> torch.dtype:
    # You said you're on a 3090; bf16 is usually fine. We still add a safe fallback.
    preferred = torch.bfloat16
    if torch.cuda.is_available():
        try:
            # Simple check path: try allocating a tiny bf16 tensor on GPU.
            _ = torch.zeros(1, device="cuda", dtype=preferred)
            return preferred
        except Exception:
            pass
        # fallback to fp16 on GPU
        return torch.float16
    # CPU fallback (bf16 on CPU is also possible, but slower; use fp32 for safety)
    return torch.float32

def main() -> None:
    dtype = pick_dtype()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device} with dtype {dtype}...")

    # NOTE: Gemma 2 (2B) may require a Hugging Face token with license acceptance.
    # If you see an auth error, run `huggingface-cli login` first.
    model = ReplacementModel.from_pretrained(
        "google/gemma-2-2b",
        "gemma",
        dtype=dtype
    )

    # ---------------------------
    # Dallas -> Austin example
    # ---------------------------
    dallas_austin_url = (
        "https://www.neuronpedia.org/gemma-2-2b/graph?slug=gemma-fact-dallas-austin"
        "&clerps=%5B%5D&pruningThreshold=0.53"
        "&pinnedIds=27_22605_10%2C20_15589_10%2CE_26865_9%2C21_5943_10%2C23_12237_10%2C20_15589_9"
        "%2C16_25_9%2C14_2268_9%2C18_8959_10%2C4_13154_9%2C7_6861_9%2C19_1445_10%2CE_2329_7%2CE_6037_4"
        "%2C0_13727_7%2C6_4012_7%2C17_7178_10%2C15_4494_4%2C6_4662_4%2C4_7671_4%2C3_13984_4%2C1_1000_4"
        "%2C19_7477_9%2C18_6101_10%2C16_4298_10%2C7_691_10"
        "&supernodes=%5B%5B%22capital%22%2C%2215_4494_4%22%2C%226_4662_4%22%2C%224_7671_4%22%2C%223_13984_4%22"
        "%2C%221_1000_4%22%5D%2C%5B%22state%22%2C%226_4012_7%22%2C%220_13727_7%22%5D%2C%5B%22Texas%22%2C%2220_15589_9%22"
        "%2C%2219_7477_9%22%2C%2216_25_9%22%2C%224_13154_9%22%2C%2214_2268_9%22%2C%227_6861_9%22%5D%2C%5B%22preposition"
        "+followed+by+place+name%22%2C%2219_1445_10%22%2C%2218_6101_10%22%5D%2C%5B%22capital+cities+%2F+say+a+capital+city%22"
        "%2C%2221_5943_10%22%2C%2217_7178_10%22%2C%227_691_10%22%2C%2216_4298_10%22%5D%5D"
    )
    supernode_features = extract_supernode_features(dallas_austin_url)

    # Supernodes that upweight certain outputs
    say_austin_node = Supernode(
        name="Say Austin",
        features=[Feature(layer=23, pos=10, feature_idx=12237)]
    )
    say_capital_node = Supernode(
        name="Say a capital",
        features=supernode_features["capital cities / say a capital city"],
        children=[say_austin_node],
    )

    # Intermediate nodes
    texas_node = Supernode(
        name="Texas",
        features=supernode_features["Texas"],
        children=[say_austin_node],
    )
    state_node = Supernode(
        name="state",
        features=supernode_features["state"],
        children=[say_capital_node, texas_node],
    )
    capital_node = Supernode(
        name="capital",
        features=supernode_features["capital"],
        children=[say_capital_node],
    )

    # Embedding nodes (no features, just edges in the visualization)
    dallas_node = Supernode(name="Emb: Dallas", features=None, children=[texas_node])
    state_emb_node = Supernode(name="Emb: state", features=None, children=[state_node])
    capital_emb_node = Supernode(name="Emb: capital", features=None, children=[capital_node])

    prompt = "Fact: the capital of the state containing Dallas is"
    ordered_nodes = [
        [capital_emb_node, state_emb_node],
        [capital_node, state_node, dallas_node],
        [say_capital_node, texas_node],
        [say_austin_node],
    ]
    dallas_austin_graph = InterventionGraph(ordered_nodes=ordered_nodes, prompt=prompt)

    # Get logits + activations
    logits, dallas_activations = model.get_activations(prompt)

    # Initialize nodes with default activations
    for node in [capital_node, state_node, dallas_node, say_capital_node, texas_node, say_austin_node]:
        dallas_austin_graph.initialize_node(node, dallas_activations)

    # Set current activations to 100% of default
    dallas_austin_graph.set_node_activation_fractions(dallas_activations)

    # Utility: top-k outputs
    def get_top_outputs(logits_tensor: torch.Tensor, k: int = 5):
        top_probs, top_token_ids = logits_tensor.squeeze(0)[-1].softmax(-1).topk(k)
        top_tokens = [model.tokenizer.decode(token_id) for token_id in top_token_ids]
        return list(zip(top_tokens, top_probs.tolist()))

    top_outputs = get_top_outputs(logits)

    # Initial graph
    html0 = create_graph_visualization(dallas_austin_graph, top_outputs)
    save_html(html0, THIS_DIR / "outputs" / "00_initial_graph.html")

    # ---------------------------
    # First intervention: ablate "Say a capital" as per the demo
    # ---------------------------
    Intervention = namedtuple("Intervention", ["supernode", "scaling_factor"])

    def supernode_intervention(
        intervention_graph: InterventionGraph,
        interventions: List[Intervention],
        replacements: Optional[Dict[str, Supernode]] = None,
    ):
        """Perform interventions on a set of supernodes and return an HTML viz."""
        # Build (layer, pos, feature_idx, new_value) tuples
        intervention_values = []
        for intervened_supernode, scaling_factor in interventions:
            if intervened_supernode.features is None or intervened_supernode.default_activations is None:
                # no-op if a node has no features (like embedding placeholders)
                continue
            for feature, default_act in zip(intervened_supernode.features, intervened_supernode.default_activations):
                intervention_values.append((*feature, scaling_factor * default_act))

        new_logits, new_activations = model.feature_intervention(intervention_graph.prompt, intervention_values)
        intervention_graph.set_node_activation_fractions(new_activations)
        top_outputs_local = get_top_outputs(new_logits)

        # annotate nodes for the visualization
        for intervened_supernode, scaling_factor in interventions:
            intervened_supernode.activation = None
            intervened_supernode.intervention = f"{scaling_factor}x"

        if replacements is not None:
            for target, replacement in replacements.items():
                intervention_graph.nodes[target].replacement_node = replacement

        return create_graph_visualization(intervention_graph, top_outputs_local)

    html1 = supernode_intervention(dallas_austin_graph, [Intervention(say_capital_node, -2)])
    save_html(html1, THIS_DIR / "outputs" / "01_ablate_say_capital.html")

    print("\nDone. Open the HTML files in ./outputs/")
    print("   - 00_initial_graph.html")
    print("   - 01_ablate_say_capital.html")

if __name__ == "__main__":
    main()
