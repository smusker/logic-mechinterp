# austin_circuit_trace.py
# Run this as a normal Python script (not a notebook).
# It will save HTML visualizations to disk (see prints at the end).
# This version pulls the pre-constructed graph from Neuronpedia and uses your
# supernode groups: triangle, square, round, decision, say_no (with the given children).

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
    # On a 3090, bf16 usually works; we still add a safe fallback.
    preferred = torch.bfloat16
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda", dtype=preferred)
            return preferred
        except Exception:
            return torch.float16
    return torch.float32

def main() -> None:
    dtype = pick_dtype()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device} with dtype {dtype}...")

    # IMPORTANT: Use the SAME base model that your Neuronpedia graph was built on.
    # Set these via env vars if needed:
    #   export CT_MODEL_ID="Qwen/<exact-model-id>"
    #   export CT_MODEL_FAMILY="qwen"

    """
    MODEL_ID = os.environ.get("CT_MODEL_ID", "Qwen/Qwen3-4B")   # <-- adjust to match your graph
    MODEL_FAMILY = os.environ.get("CT_MODEL_FAMILY", "qwen")      

    model = ReplacementModel.from_pretrained(
        MODEL_ID,
        MODEL_FAMILY,
        dtype=dtype
    )

    """

    model = ReplacementModel.from_pretrained(
        model_name="Qwen/Qwen3-4B",
        transcoder_set="mwhanna/qwen3-4b-transcoders", #note that these are per layer transcoders not cross layer transcoders. Not sure if that's good enough. 
        #dtype="bfloat16",
        offload="cpu",
    )

    # ---------------------------
    # Your new prompt + Neuronpedia graph
    # ---------------------------
    prompt = (
        "Label true if the object is round. Then give an explanation.\n\n"
        "square : false\n"
        "circle : true\n\n"
        "In one word, should triangle be labelled true?\n\n"
    )

    neuronpedia_url = ("https://www.neuronpedia.org/qwen3-4b/graph?slug=labeltrueiftheob-1754863104323&pruningThreshold=0.6&densityThreshold=1&pinnedIds=37_2753_33%2C8_40987_9%2C5_47794_9%2C1_3148_9%2CE_4778_9%2C28_35916_29%2C27_39240_29%2C24_18550_29%2C10_110454_29%2C8_59778_29%2C5_15525_29%2C3_15633_29%2C0_5048_29%2C0_20747_29%2C3_16128_29%2C5_50991_29%2C3_48472_29%2CE_21495_29%2C30_28614_33%2C18_77725_33%2C22_76459_33%2C33_32581_33%2C26_67146_33%2C31_124849_33%2C18_9725_29%2C0_111654_16%2C0_92505_16%2C0_44724_16%2C0_31909_16&supernodes=%5B%5B%22round%22%2C%221_3148_9%22%2C%225_47794_9%22%2C%228_40987_9%22%5D%2C%5B%22triangle%22%2C%223_15633_29%22%2C%2210_110454_29%22%2C%2227_39240_29%22%2C%2224_18550_29%22%2C%2228_35916_29%22%2C%220_5048_29%22%2C%228_59778_29%22%2C%225_15525_29%22%2C%223_16128_29%22%2C%220_20747_29%22%2C%225_50991_29%22%2C%223_48472_29%22%5D%2C%5B%22square%22%2C%220_31909_16%22%2C%220_44724_16%22%2C%220_92505_16%22%5D%2C%5B%22say_no%22%2C%2233_32581_33%22%2C%2226_67146_33%22%2C%2231_124849_33%22%5D%2C%5B%22decision%22%2C%2222_76459_33%22%2C%220_111654_16%22%2C%2230_28614_33%22%2C%2218_77725_33%22%2C%2218_9725_29%22%5D%5D"
    )

    # Pull the supernode features by group name from Neuronpedia
    supernode_features = extract_supernode_features(neuronpedia_url)

    # Build Supernodes to match your grouping + children
    triangle_node = Supernode(
        name="triangle",
        features=supernode_features["triangle"],  # list[Feature]
        children=[]  # children set below to avoid forward ref issues
    )
    square_node = Supernode(
        name="square",
        features=supernode_features["square"],
        children=[]
    )
    round_node = Supernode(
        name="round",
        features=supernode_features["round"],
        children=[]
    )
    decision_node = Supernode(
        name="decision",
        features=supernode_features["decision"],
        children=[]
    )
    say_no_node = Supernode(
        name="say_no",
        features=supernode_features["say_no"],
        children=[]
    )

    # Now wire up the children exactly as you specified

    # Supernode: "triangle" -> children: decision, say_no, round, square
    triangle_node.children = [decision_node, say_no_node, round_node, square_node]

    # Supernode: "square" -> children: decision, say_no, triangle
    square_node.children = [decision_node, say_no_node, triangle_node]

    # Supernode: "round" -> children: decision, say_no, triangle
    round_node.children = [decision_node, say_no_node, triangle_node]

    # Supernode: "decision" -> children: triangle, square, round, say_no
    decision_node.children = [triangle_node, square_node, round_node, say_no_node]

    # Supernode: "say_no" -> (no children listed)
    say_no_node.children = []

    round_emb_node = Supernode(name="Emb: round", features=None, children=[triangle_node, square_node, round_node, decision_node, say_no_node])
    triangle_emb_node = Supernode(name="Emb: triangle", features=None, children=[triangle_node, decision_node, say_no_node])

    # Choose a simple topological-ish order for visualization
    ordered_nodes = [
        [round_emb_node, triangle_emb_node], 
        [triangle_node, square_node, round_node],
        [decision_node],
        [say_no_node],
    ]

    graph = InterventionGraph(ordered_nodes=ordered_nodes, prompt=prompt)

    # Get logits + activations for the prompt
    logits, activations = model.get_activations(prompt)

    # Initialize nodes with default activations (only nodes with features)
    for node in [triangle_node, square_node, round_node, decision_node, say_no_node]:
        graph.initialize_node(node, activations)

    # Set current activations to 100% of default
    graph.set_node_activation_fractions(activations)

    # Utility: top-k outputs
    def get_top_outputs(logits_tensor: torch.Tensor, k: int = 5):
        top_probs, top_token_ids = logits_tensor.squeeze(0)[-1].softmax(-1).topk(k)
        top_tokens = [model.tokenizer.decode(token_id) for token_id in top_token_ids]
        return list(zip(top_tokens, top_probs.tolist()))

    top_outputs = get_top_outputs(logits)

    output_sub_directory = "triangle_round"

    # Save baseline graph
    html0 = create_graph_visualization(graph, top_outputs)
    save_html(html0, THIS_DIR / "outputs" / output_sub_directory / "00_initial_graph.html")

    # ---------------------------
    # Interventions
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

    # A. Ablate "round" (downweight evidence for the roundness concept)
    html1 = supernode_intervention(graph, [Intervention(round_node, -2)])
    save_html(html1, THIS_DIR / "outputs" / output_sub_directory / "01_ablate_round.html")

    # B. Boost "round" (upweight roundness evidence)
    html2 = supernode_intervention(graph, [Intervention(round_node, 2)])
    save_html(html2, THIS_DIR / "outputs" / output_sub_directory / "02_boost_round.html")

    # C. Ablate "decision" node
    html3 = supernode_intervention(graph, [Intervention(decision_node, -2)])
    save_html(html3, THIS_DIR / "outputs" / output_sub_directory / "03_ablate_decision.html")

    print("\nDone. Open the HTML files in ./outputs/"+str(output_sub_directory))
    print("   - 00_initial_graph.html")
    print("   - 01_ablate_round.html")
    print("   - 02_boost_round.html")
    print("   - 03_ablate_decision.html")
    print("\nTip: set CT_MODEL_ID/CT_MODEL_FAMILY to the exact base model used in your Neuronpedia graph.")

if __name__ == "__main__":
    main()
