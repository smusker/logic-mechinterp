# xor_boundless_das_llama3.py
# =====================================================================
#  Experiment: Boundless DAS on Meta-Llama-3-8B-Instruct
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
)

# ──────────────────────────────────────────────────────────────────────
#  xor-specific helpers
# ──────────────────────────────────────────────────────────────────────
from xor_utils import (
    build_loader,
    cf_pair_model_A,
    cf_pair_model_B,
    sample_examples,
    random_obj,
    make_prompt,
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
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

# ── Label-token IDs we’ll always use ──────────────────────────────────
TRUE_ID  = tokenizer.encode("True",  add_special_tokens=False)[0]
FALSE_ID = tokenizer.encode("False", add_special_tokens=False)[0]
LABEL_IDS = torch.tensor([TRUE_ID, FALSE_ID], device=device)
print("ID for un-spaced 'True' :", TRUE_ID)
print("ID for un-spaced 'False':", FALSE_ID)


assert tokenizer.decode([TRUE_ID])  == "True"
assert tokenizer.decode([FALSE_ID]) == "False"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16
).to(device)
model.eval()

print(f"Model hidden size: {model.config.hidden_size}")

# ── QUICK SANITY: every abbreviation must be ONE token ────────────────
from xor_utils import CO_ABR, SH_ABR
for word, abr in {**CO_ABR, **SH_ABR}.items():
    ids = tokenizer(" " + abr, add_special_tokens=False).input_ids
    assert len(ids) == 1, (
        f"Abbreviation '{abr}' for '{word}' splits into {len(ids)} tokens → "
        "prompt length will drift!"
    )
print("✓ all colour/shape abbreviations are single tokens.")


"""

print(tokenizer(" True",  add_special_tokens=False).input_ids)
print(tokenizer(" False", add_special_tokens=False).input_ids)


print(tokenizer("True",  add_special_tokens=False).input_ids)
print(tokenizer("False", add_special_tokens=False).input_ids)

"""

# 1 b ─────────────────────────────────────────────────────────────────
#  compute the constant prompt length & token-to-probe index
# --------------------------------------------------------------------
query_obj   = random_obj()
dummy_ids, _ = make_prompt(tokenizer,
                           sample_examples(exclude=query_obj),  # ← added
                           query_obj)
SEQ_LEN        = dummy_ids.size(0)
TOKEN_TO_PROBE = SEQ_LEN - 2        # penultimate token (right before label)
print(f"Prompt length = {SEQ_LEN} tokens; probing token index {TOKEN_TO_PROBE}")

# ── CHECK prompt-length invariance over a few random draws ────────────
for _ in range(20):
    q = random_obj()
    ids, _ = make_prompt(tokenizer, sample_examples(exclude=q), q)
    assert ids.size(0) == SEQ_LEN, "Prompt length drift detected!"
print("✓ prompt length stable across random samples.")


# 1 c ─────────────────────────────────────────────────────────────────
#  quick factual-accuracy check (no interventions)
# --------------------------------------------------------------------
def build_factual_loader(tok, n=200, batch_size=4):
    bases, labels = [], []
    for _ in range(n):
        q = random_obj()
        ids, lbl = make_prompt(tok, sample_examples(exclude=q), q)
        bases.append(ids.tolist()); labels.append(lbl.tolist())
    from datasets import Dataset
    from torch.utils.data import DataLoader
    ds = Dataset.from_dict({"input_ids": bases, "labels": labels}).with_format("torch")
    return DataLoader(ds, batch_size=batch_size)

@torch.no_grad()
def compute_accuracy(llm, loader):
    total = correct = 0
    for batch in loader:
        for k, v in batch.items():
            batch[k] = v.to(device)
        out = llm(input_ids=batch["input_ids"])          # [B, L, V]
        two_logits = out.logits[:, -1, LABEL_IDS]        # [B, 2]
        pred_is_true = two_logits.argmax(-1) == 0        # 0=True, 1=False
        gold_is_true = batch["labels"][:, -1] == TRUE_ID
        correct += (pred_is_true == gold_is_true).sum().item()
        total   += batch["input_ids"].size(0)
    return correct / total

factual_loader = build_factual_loader(tokenizer)

# ── INSPECT one sample prompt + model prediction ─────────────────────
dbg_q = random_obj()
ids_dbg, lbl_dbg = make_prompt(tokenizer, sample_examples(exclude=dbg_q), dbg_q)
print("\n──────── SAMPLE PROMPT (decoded) ────────")
print(tokenizer.decode(ids_dbg))
print("Gold label token id :", lbl_dbg[-1].item(),
      "| text:", tokenizer.decode([lbl_dbg[-1].item()]))

with torch.no_grad():
    out_dbg = model(input_ids=ids_dbg.unsqueeze(0).to(device))
pred_id = int(out_dbg.logits[0, -1].argmax())
print("Model prediction id :", pred_id,
      "| text:", tokenizer.decode([pred_id]))
print("──────── END SAMPLE ────────\n")

# Also print the two label tokens the scorer cares about
print("ID for ' True' :", tokenizer.encode(' True',  add_special_tokens=False)[0])
print("ID for ' False':", tokenizer.encode(' False', add_special_tokens=False)[0])



fact_acc = compute_accuracy(model, factual_loader)
print(f"Factual task accuracy (no swaps) : {fact_acc:.2%}")


# 2 ────────────────────────────────────────────────────────────────────
#  build counter-factual datasets
# ---------------------------------------------------------------------
train_A = build_loader(tokenizer, cf_pair_model_A, 8000, batch_size=4)
test_A  = build_loader(tokenizer, cf_pair_model_A, 1000, batch_size=4, shuffle=False)

train_B = build_loader(tokenizer, cf_pair_model_B, 8000, batch_size=4)
test_B  = build_loader(tokenizer, cf_pair_model_B, 1000, batch_size=4, shuffle=False)

# 3 ────────────────────────────────────────────────────────────────────
#  Build IntervenableModel  (layer 12, penultimate token)
# ---------------------------------------------------------------------
LAYER_TO_PROBE = 12

def make_intervenable():
    cfg = IntervenableConfig(
        model_type         = type(model),
        representations    = [RepresentationConfig(LAYER_TO_PROBE, "block_output")],
        intervention_types = BoundlessRotatedSpaceIntervention,
    )
    iv = IntervenableModel(cfg, model)
    iv.set_device(device)
    iv.disable_model_gradients()
    return iv

# 4 ────────────────────────────────────────────────────────────────────
#  generic trainer
# ---------------------------------------------------------------------
def train_das(intervenable, loader, epochs=2):
    intv   = next(iter(intervenable.interventions.values()))
    optim  = torch.optim.Adam(
        [{"params": intv.rotate_layer.parameters()},
         {"params": intv.intervention_boundaries, "lr":1e-2}],
        lr=1e-3,
    )
    total_steps = epochs * len(loader)
    temp_sched  = torch.linspace(50.0, 0.1, total_steps, device=device)
    step = 0
    for ep in range(epochs):
        pbar = tqdm(loader, desc=f"epoch {ep}")
        for batch in pbar:
            for k,v in batch.items(): batch[k] = v.to(device)
            optim.zero_grad()
            _, cf = intervenable(
                {"input_ids": batch["input_ids"]},
                [{"input_ids": batch["source_input_ids"]}],
                {"sources->base": TOKEN_TO_PROBE},
            )
            # keep only the 2-way logits (True vs False) for the last position
            two_logits = cf.logits[:, -1, LABEL_IDS]          # [B, 2]
            gold_index = (batch["labels"][:, -1] == FALSE_ID).long()  # 0=True, 1=False
            loss = CrossEntropyLoss()(two_logits, gold_index)

            loss += intv.intervention_boundaries.sum()
            loss.backward(); optim.step()
            intv.set_temperature(temp_sched[step]); step += 1
            pbar.set_postfix({"loss": round(loss.item(),2)})

# 5 ────────────────────────────────────────────────────────────────────
#  IIA computation
# ---------------------------------------------------------------------
@torch.no_grad()
def compute_iia(intervenable, loader):
    total = correct = 0
    for batch in loader:
        for k, v in batch.items():
            batch[k] = v.to(device)
        _, cf = intervenable(
            {"input_ids": batch["input_ids"]},
            [{"input_ids": batch["source_input_ids"]}],
            {"sources->base": TOKEN_TO_PROBE},
        )
        two_logits = cf.logits[:, -1, LABEL_IDS]            # [B, 2]
        pred_is_true = two_logits.argmax(-1) == 0
        gold_is_true = batch["labels"][:, -1] == TRUE_ID
        correct += (pred_is_true == gold_is_true).sum().item()
        total   += batch["input_ids"].size(0)
    return correct / total


# 6 ────────────────────────────────────────────────────────────────────
#  run Model A
# ---------------------------------------------------------------------
"""
iv_A = make_intervenable()
print("\n⫸  Training alignment for causal Model A (single XOR)")
train_das(iv_A, train_A, epochs=2)
iia_A = compute_iia(iv_A, test_A)
print(f"IIA (Model A) = {iia_A:.2%}")
"""

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
