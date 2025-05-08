# xor_boundless_das.py
# ---------------------------------------------------------------------
# identical imports to the paper’s tutorial
# ---------------------------------------------------------------------
import torch, random, numpy as np, os
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
from transformers import set_seed

# pyvene helpers
from pyvene import (
    create_llama,
    IntervenableModel,
    RepresentationConfig,
    IntervenableConfig,
    BoundlessRotatedSpaceIntervention,
)
from pyvene import count_parameters

# local data utils
from xor_utils import (
    build_loader, cf_pair_model_A, cf_pair_model_B
)

# reproducibility ------------------------------------------------------
set_seed(42); random.seed(42); np.random.seed(42)

# ---------------------------------------------------------------------
# load the same Alpaca/LLaMA-7B that the price-tagging notebook used
# ---------------------------------------------------------------------
cfg, tok, llama = create_llama()
llama.to("cuda"); llama.eval()

# ---------------------------------------------------------------------
# build counter-factual datasets
# ---------------------------------------------------------------------
train_A = build_loader(tok, cf_pair_model_A,  8000, batch_size=16)
eval_A  = build_loader(tok, cf_pair_model_A,  1000, batch_size=16, shuffle=False)
test_A  = build_loader(tok, cf_pair_model_A,  1000, batch_size=16, shuffle=False)

train_B = build_loader(tok, cf_pair_model_B,  8000, batch_size=16)
eval_B  = build_loader(tok, cf_pair_model_B,  1000, batch_size=16, shuffle=False)
test_B  = build_loader(tok, cf_pair_model_B,  1000, batch_size=16, shuffle=False)

# ---------------------------------------------------------------------
# create IntervenableModel (layer 12, token index −2 i.e. penultimate)
# ---------------------------------------------------------------------
LAYER  = 12
intervene_token = -2          # constant index across prompts

def build_intervenable():
    cfg_int = IntervenableConfig(
        model_type = type(llama),
        representations=[RepresentationConfig(LAYER, "block_output")],
        intervention_types = BoundlessRotatedSpaceIntervention,
    )
    m = IntervenableModel(cfg_int, llama)
    m.set_device("cuda")
    m.disable_model_gradients()
    return m

# ---------------------------------------------------------------------
# training routine borrowed verbatim from tutorial
# ---------------------------------------------------------------------
def train_boundless_das(intervenable, dataloader, epochs=3):
    intv = list(intervenable.interventions.values())[0]   # a shortcut
    params = [
        {"params": intv.rotate_layer.parameters()},
        {"params": intv.intervention_boundaries, "lr":1e-2},
    ]
    opt = torch.optim.Adam(params, lr=1e-3)
    total_steps = epochs * len(dataloader)
    warmup = int(0.1*total_steps)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1.0, end_factor=0.0, total_iters=total_steps
    )
    temp_sched = torch.linspace(50., 0.1, total_steps, device="cuda")

    step = 0
    for ep in range(epochs):
        pbar = tqdm(dataloader, desc=f"epoch {ep}")
        for batch in pbar:
            opt.zero_grad()
            for k,v in batch.items():
                batch[k]=v.to("cuda")
            _, cf = intervenable(
                    {"input_ids":batch["input_ids"]},
                    [{"input_ids":batch["source_input_ids"]}],
                    {"sources->base": intervene_token},
                 )
            loss = CrossEntropyLoss()(
                     cf.logits.view(-1, cf.logits.size(-1)),
                     batch["labels"].view(-1)
                   )
            loss += 1.0*intv.intervention_boundaries.sum()
            loss.backward(); opt.step(); scheduler.step()
            intv.set_temperature(temp_sched[step]); step+=1
            pbar.set_postfix({"loss": round(loss.item(),2)})

# ---------------------------------------------------------------------
# interchange-intervention accuracy (IIA)
# ---------------------------------------------------------------------
def compute_iia(intervenable, dataloader):
    total, correct = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            for k,v in batch.items(): batch[k]=v.to("cuda")
            _, cf = intervenable(
                  {"input_ids":batch["input_ids"]},
                  [{"input_ids":batch["source_input_ids"]}],
                  {"sources->base": intervene_token},
            )
            preds = cf.logits.argmax(-1)
            mask = (batch["labels"] != -100)
            total   += mask.sum().item()
            correct += ((preds==batch["labels"]) & mask).sum().item()
    return correct/total

# ---------------------------------------------------------------------
# 1. Model A  ----------------------------------------------------------
# ---------------------------------------------------------------------
iva = build_intervenable()
print("∎ training Model A alignment")
train_boundless_das(iva, train_A, epochs=2)
iia_A = compute_iia(iva, test_A)
print(f"IIA (Model A) = {iia_A:.2%}")

# ---------------------------------------------------------------------
# 2. Model B  ----------------------------------------------------------
# ---------------------------------------------------------------------
ivb = build_intervenable()
print("∎ training Model B alignment")
train_boundless_das(ivb, train_B, epochs=2)
iia_B = compute_iia(ivb, test_B)
print(f"IIA (Model B) = {iia_B:.2%}")

# ---------------------------------------------------------------------
# final verdict
# ---------------------------------------------------------------------
print("\n────────  RESULT  ────────")
if iia_A > iia_B: print("Network matches single-XOR causal model A.")
elif iia_B > iia_A: print("Network matches decomposed model B.")
else:              print("Tie – both abstractions fit equally well.")