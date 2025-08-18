# austin_circuit_trace.py
# Fully local: builds the attribution graph from scratch (no Neuronpedia),
# extracts features to build supernodes, then runs interventions and saves HTML.

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from collections import namedtuple
from typing import Dict, List, Optional, Tuple

import torch

# --- Paths: we expect the circuit-tracer repo to be cloned into repository/circuit-tracer ---
THIS_DIR = Path(__file__).resolve().parent
CT_ROOT = (THIS_DIR / "repository" / "circuit-tracer").resolve()
OUT_DIR = THIS_DIR / "outputs"
GRAPH_DIR = OUT_DIR / "graph_files" / "local-dallas-austin"  # CLI will write JSONs here
RAW_GRAPH_PATH = OUT_DIR / "raw_graph.pt"

if not CT_ROOT.exists():
    raise SystemExit(
        f"Could not find circuit-tracer at {CT_ROOT}.\n"
        "Clone the repo into repository/circuit-tracer (see setup commands)."
    )

# Ensure we can import the package and its demo utils
sys.path.insert(0, str(CT_ROOT))
sys.path.insert(0, str(CT_ROOT / "demos"))

from circuit_tracer import ReplacementModel  # type: ignore
from graph_visualization import create_graph_visualization, Supernode, InterventionGraph, Feature  # type: ignore

# ---------------------------
# Helper: save HTML objects
# ---------------------------
def save_html(obj, path: Path) -> None:
    """Save an IPython.display.HTML object or raw HTML string to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    html_text = getattr(obj, "data", None)
    if html_text is None:
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
            _ = torch.zeros(1, device="cuda", dtype=preferred)
            return preferred
        except Exception:
            return torch.float16
    return torch.float32

# -------------------------------------------------------
# Step 1: Build the attribution graph locally (via CLI)
# -------------------------------------------------------
def run_cli_attribution(
    prompt: str,
    slug: str,
    graph_file_dir: Path,
    raw_graph_path: Path,
    transcoder_set: str = "gemma",
    node_threshold: float = 0.53,   # close to the notebook/URL pruning value
    edge_threshold: float = 0.98,
    dtype_flag: str = "bf16",       # try bf16; library also accepts fp16/fp32; see README
) -> None:
    """
    Uses the official CLI to compute attribution + emit JSON graph files.
    """
    exe = shutil.which("circuit-tracer")
    if exe is None:
        raise RuntimeError(
            "Could not find `circuit-tracer` on PATH. Did you `pip install -e ./repository/circuit-tracer`?"
        )

    graph_file_dir.mkdir(parents=True, exist_ok=True)
    raw_graph_path.parent.mkdir(parents=True, exist_ok=True)

    # Build command per README's "3-step process" (attribution -> graph files)
    # We'll NOT start the server; we only want the files.
    # Ref: README's CLI docs. :contentReference[oaicite:1]{index=1}
    cmd = [
        exe, "attribute",
        "--prompt", prompt,
        "--transcoder_set", transcoder_set,
        "--slug", slug,
        "--graph_file_dir", str(graph_file_dir.parent),   # CLI will create a subdir named <slug>
        "--graph_output_path", str(raw_graph_path),
        "--node_threshold", str(node_threshold),
        "--edge_threshold", str(edge_threshold),
        "--dtype", dtype_flag,
        "--max_n_logits", "10",
        "--desired_logit_prob", "0.95",
        "--batch_size", "256",
        "--verbose",
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Attribution complete. Graph files in: {graph_file_dir.parent} ; raw graph: {raw_graph_path}")

# -------------------------------------------------------
# Step 2: Parse emitted JSON graph files to recover features
# -------------------------------------------------------
def _load_graph_json(graph_dir: Path) -> Dict:
    """
    The CLI writes a folder named after the slug under graph_file_dir.
    Inside, there will be one or more JSON files for the viewer.
    We scan for the main graph JSON (contains 'nodes' and 'edges').
    """
    if not graph_dir.exists():
        # Some CLI versions may create the slug folder *one level up*; try parent
        graph_dir = graph_dir.parent

    candidates = list(graph_dir.rglob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No JSON files found under {graph_dir}")

    # Heuristic: find a file with both 'nodes' and 'edges' top-level keys
    for p in candidates:
        try:
            data = json.loads(p.read_text())
            if isinstance(data, dict) and "nodes" in data and "edges" in data:
                print(f"Using graph JSON: {p}")
                return data
        except Exception:
            continue

    # Fallback: return the largest JSON file (likely the graph)
    p = max(candidates, key=lambda x: x.stat().st_size)
    print(f"Using (fallback) graph JSON: {p}")
    return json.loads(p.read_text())

def _normalize_node(n: Dict) -> Dict:
    """
    Normalize different possible viewer JSON shapes into a common schema:
    {
      'id': str,
      'type': 'feature' | 'token' | 'logit' | ...,
      'label': str,
      'layer': int | None,
      'pos': int | None,
      'feature_idx': int | None,
      'meta': dict
    }
    """
    node = {
        "id": n.get("id") or n.get("_id") or n.get("uid") or str(n.get("key") or n.get("name")),
        "type": n.get("type") or n.get("node_type") or n.get("kind"),
        "label": n.get("label") or n.get("name") or n.get("text") or "",
        "layer": None,
        "pos": None,
        "feature_idx": None,
        "meta": n,
    }

    # Common places the tuple shows up
    for k in ("feature", "feature_idx", "feat_idx"):
        if isinstance(n.get(k), (tuple, list)) and len(n[k]) == 3:
            node["layer"], node["pos"], node["feature_idx"] = map(int, n[k])

    # Or sometimes embedded separately
    for a, b in (("layer", "layer"), ("position", "pos"), ("pos", "pos"), ("feature_index", "feature_idx")):
        if n.get(a) is not None:
            node[b] = int(n[a])

    return node

def _find_features_by_label(
    nodes: List[Dict],
    include_patterns: List[str],
    exclude_patterns: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> List[Tuple[int, int, int]]:
    """
    Return a list of (layer, pos, feature_idx) for nodes whose labels match
    any of the include_patterns (case-insensitive), excluding any exclude_patterns.
    Only returns entries for which we can recover the full feature triple.
    """
    inc_res = [re.compile(pat, re.I) for pat in include_patterns]
    exc_res = [re.compile(pat, re.I) for pat in (exclude_patterns or [])]

    matches: List[Tuple[int, int, int]] = []
    for n in nodes:
        if n["type"] not in ("feature", "transcoder_feature", "mlp_feature"):
            continue
        label = n["label"] or ""
        if not any(r.search(label) for r in inc_res):
            continue
        if exc_res and any(r.search(label) for r in exc_res):
            continue
        if n["layer"] is None or n["pos"] is None or n["feature_idx"] is None:
            continue
        matches.append((n["layer"], n["pos"], n["feature_idx"]))

    if limit is not None:
        return matches[:limit]
    return matches

def build_supernode_feature_map_from_local_graph(prompt: str) -> Dict[str, List[Feature]]:
    """
    1) Runs CLI attribution to create JSON graph files for `prompt`.
    2) Parses the JSON to recover features for a set of supernodes we care about
       in the Dallas->Austin example: 'capital', 'state', 'Texas',
       'capital cities / say a capital city', and 'Say Austin'.

    Returns a dict[str, list[Feature]] compatible with our Supernode building.
    """
    # 1) Build graph (idempotent-ish; re-run will overwrite files)
    run_cli_attribution(
        prompt=prompt,
        slug="local-dallas-austin",
        graph_file_dir=GRAPH_DIR,
        raw_graph_path=RAW_GRAPH_PATH,
        transcoder_set="gemma",
        node_threshold=0.53,
        edge_threshold=0.98,
        dtype_flag="bf16",
    )

    # 2) Parse JSON and extract features
    data = _load_graph_json(GRAPH_DIR)
    raw_nodes = data["nodes"] if isinstance(data["nodes"], list) else list(data["nodes"].values())
    nodes = [_normalize_node(n) for n in raw_nodes]

    # Heuristic label matchers that mirror the Neuronpedia-labeled supernodes:
    # - "capital" supernode (general concept)
    capital_feats = _find_features_by_label(nodes, include_patterns=[r"\bcapital\b"])
    # - "state" supernode (avoid 'United States')
    state_feats = _find_features_by_label(nodes, include_patterns=[r"\bstate\b"], exclude_patterns=[r"united\s+states"])
    # - "Texas" (the intermediate state node)
    texas_feats = _find_features_by_label(nodes, include_patterns=[r"\btexas\b"])
    # - "capital cities / say a capital city" (various phrasings)
    say_capital_feats = _find_features_by_label(
        nodes,
        include_patterns=[r"capital cities", r"say a capital", r"capital.*(say|output)"],
    )
    # - "Say Austin": sometimes labeled as a 'Say <token>' feature or similar; also try the token itself
    say_austin_feats = _find_features_by_label(
        nodes,
        include_patterns=[r"\b(austin)\b", r"\bsay\s+austin\b", r"output.*austin"],
    )

    # Convert to Feature tuples for the visualization code
    def to_Features(triples: List[Tuple[int, int, int]]) -> List[Feature]:
        return [Feature(layer=l, pos=p, feature_idx=f) for (l, p, f) in triples]

    feature_map = {
        "capital": to_Features(capital_feats),
        "state": to_Features(state_feats),
        "Texas": to_Features(texas_feats),
        "capital cities / say a capital city": to_Features(say_capital_feats),
        "Say Austin": to_Features(say_austin_feats[:1]) or [Feature(layer=23, pos=10, feature_idx=12237)],  # fallback
    }

    # Light sanity printout so you can eyeball counts
    print("Recovered features:")
    for k, v in feature_map.items():
        print(f"  {k:40s}  {len(v)}")
    return feature_map

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
    # Dallas -> Austin, built LOCALLY
    # ---------------------------
    prompt = "Fact: the capital of the state containing Dallas is"
    supernode_features = build_supernode_feature_map_from_local_graph(prompt)

    # Supernodes (mirror the notebook structure)
    say_austin_node = Supernode(
        name="Say Austin",
        features=supernode_features["Say Austin"]
    )
    say_capital_node = Supernode(
        name="Say a capital",
        features=supernode_features["capital cities / say a capital city"],
        children=[say_austin_node],
    )

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

    # Embedding nodes (edges only, no features)
    dallas_node = Supernode(name="Emb: Dallas", features=None, children=[texas_node])
    state_emb_node = Supernode(name="Emb: state", features=None, children=[state_node])
    capital_emb_node = Supernode(name="Emb: capital", features=None, children=[capital_node])

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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    html0 = create_graph_visualization(dallas_austin_graph, top_outputs)
    save_html(html0, OUT_DIR / "00_initial_graph_local.html")

    # ---------------------------
    # Intervention: ablate "Say a capital"
    # ---------------------------
    Intervention = namedtuple("Intervention", ["supernode", "scaling_factor"])

    def supernode_intervention(
        intervention_graph: InterventionGraph,
        interventions: List[Intervention],
        replacements: Optional[Dict[str, Supernode]] = None,
    ):
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

        for intervened_supernode, scaling_factor in interventions:
            intervened_supernode.activation = None
            intervened_supernode.intervention = f"{scaling_factor}x"

        if replacements is not None:
            for target, replacement in replacements.items():
                intervention_graph.nodes[target].replacement_node = replacement

        return create_graph_visualization(intervention_graph, top_outputs_local)

    html1 = supernode_intervention(dallas_austin_graph, [Intervention(say_capital_node, -2)])
    save_html(html1, OUT_DIR / "01_ablate_say_capital_local.html")

    print("\nDone. Open the HTML files in ./outputs/")
    print("   - 00_initial_graph_local.html")
    print("   - 01_ablate_say_capital_local.html")
    print(f"Graph JSONs: {GRAPH_DIR} (and possibly {GRAPH_DIR.parent})")

if __name__ == "__main__":
    main()
