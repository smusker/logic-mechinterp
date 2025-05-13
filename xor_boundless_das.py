# xor_boundless_das_llama3.py
# =====================================================================
#  Experiment:  Boundless DAS on Meta-Llama-3-8B-Instruct
# =====================================================================

import os, random, numpy as np, torch
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)

from pyvene import (
    IntervenableModel,
    RepresentationConfig,
    IntervenableConfig,
    BoundlessRotatedSpaceIntervention,
    count_parameters,
)

from xor_utils import (
    build_loader,
    cf_pair_model_A,
    cf_pair_model_B,
)

# 0 ────────────────────────────────────────────────────────────────────
#  reproducibility
# ---------------------------------------------------------------------
set_seed(42); random.seed(42); np.random.seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1 ────────────────────────────────────────────────────────────────────
#  load Llama-3-8B-Instruct
# ---------------------------------------------------------------------
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:               # add pad token if missing
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,               # bf16 fits on 24-GB GPU
    device_map="auto",
)
model.eval()

print(f"Model hidden size: {model.config.hidden_size}")

# 2 ────────────────────────────────────────────────────────────────────
#  build counter-factual datasets
# ---------------------------------------------------------------------
train_A = build_loader(tokenizer, cf_pair_model_A,  8000, batch_size=16)
eval_A  = build_loader(tokenizer, cf_pair_model_A,  1000, batch_size=16, shuffle=False)
test_A  = build_loader(tokenizer, cf_pair_model_A,  1000, batch_size=16, shuffle=False)

train_B = build_loader(tokenizer, cf_pair_model_B,  8000, batch_size=16)
eval_B  = build_loader(tokenizer, cf_pair_model_B,  1000, batch_size=16, shuffle=False)
test_B  = build_loader(tokenizer, cf_pair_model_B,  1000, batch_size=16, shuffle=False)

# 3 ────────────────────────────────────────────────────────────────────
#  Build IntervenableModel  (layer 12, penultimate token)
# ---------------------------------------------------------------------
LAYER_TO_PROBE   = 12
TOKEN_TO_PROBE   = -2            # constant across fixed-length prompt

def make_intervenable():
    cfg_intv = IntervenableConfig(
        model_type          = type(model),          # IMPORTANT CHANGE
        representations     = [RepresentationConfig(LAYER_TO_PROBE, "block_output")],
        intervention_types  = BoundlessRotatedSpaceIntervention,
    )
    iv = IntervenableModel(cfg_intv, model)
    iv.set_device(device)
    iv.disable_model_gradients()
    return iv

# 4 ────────────────────────────────────────────────────────────────────
#  generic trainer (same as tutorial)
# ---------------------------------------------------------------------
def train_das(intervenable, dataloader, epochs=2):
    intv = list(intervenable.interventions.values())[0]
    optim = torch.optim.Adam(
        [
            {"params": intv.rotate_layer.parameters()},
            {"params": intv.intervention_boundaries, "lr":1e-2},
        ],
        lr=1e-3,
    )
    total_steps = epochs * len(dataloader)
    temp_sched  = torch.linspace(50.0, 0.1, total_steps, device=device)
    step = 0

    for ep in range(epochs):
        pbar = tqdm(dataloader, desc=f"epoch {ep}")
        for batch in pbar:
            for k,v in batch.items(): batch[k] = v.to(device)
            optim.zero_grad()

            _, cf = intervenable(
                {"input_ids": batch["input_ids"]},
                [{"input_ids": batch["source_input_ids"]}],
                {"sources->base": TOKEN_TO_PROBE},
            )
            loss = CrossEntropyLoss()(
                    cf.logits.view(-1, cf.logits.size(-1)),
                    batch["labels"].view(-1))
            loss += intv.intervention_boundaries.sum()   # small L1
            loss.backward(); optim.step()
            intv.set_temperature(temp_sched[step]); step += 1
            pbar.set_postfix({"loss": round(loss.item(),2)})

# 5 ────────────────────────────────────────────────────────────────────
#  IIA computation
# ---------------------------------------------------------------------
@torch.no_grad()
def compute_iia(intervenable, dataloader):
    total = correct = 0
    for batch in dataloader:
        for k,v in batch.items(): batch[k]=v.to(device)
        _, cf = intervenable(
            {"input_ids": batch["input_ids"]},
            [{"input_ids": batch["source_input_ids"]}],
            {"sources->base": TOKEN_TO_PROBE},
        )
        pred = cf.logits.argmax(-1)
        mask = batch["labels"] != -100
        total   += mask.sum().item()
        correct += ((pred == batch["labels"]) & mask).sum().item()
    return correct / total

# 6 ────────────────────────────────────────────────────────────────────
#  run Model A
# ---------------------------------------------------------------------
iv_A = make_intervenable()
print("\n⫸  Training alignment for causal Model A (single XOR)")
train_das(iv_A, train_A, epochs=2)
iia_A = compute_iia(iv_A, test_A)
print(f"IIA (Model A) = {iia_A:.2%}")

# 7 ────────────────────────────────────────────────────────────────────
#  run Model B
# ---------------------------------------------------------------------
iv_B = make_intervenable()
print("\n⫸  Training alignment for causal Model B (decomposed)")
train_das(iv_B, train_B, epochs=2)
iia_B = compute_iia(iv_B, test_B)
print(f"IIA (Model B) = {iia_B:.2%}")

# 8 ────────────────────────────────────────────────────────────────────
#  verdict
# ---------------------------------------------------------------------
print("\n──────────  RESULT  ──────────")
if iia_A > iia_B:
    print("► The network’s internal computation matches SINGLE-XOR Model A.")
elif iia_B > iia_A:
    print("► The network’s internal computation matches DECOMPOSED Model B.")
else:
    print("► Tie – both causal abstractions fit equally well.")