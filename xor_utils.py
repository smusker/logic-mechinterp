# xor_utils.py
# --------------------------------------------------------------------------------
# Helper functions that play the same role as price_tagging_utils.py in the paper
# --------------------------------------------------------------------------------
import random, torch
from typing import List, Tuple
IGNORE = -100          # label padding

# colours / shapes used in this experiment
COLOURS = ["yellow", "blue"]
SHAPES  = ["circle", "triangle"]

CO_ABR = {"yellow": "yel", "blue": "blu"}
SH_ABR = {"circle": "cir", "triangle": "tri"}

# ---------------------------------------------------------------------
# 1. ground-truth rule and low-level feature tests
# ---------------------------------------------------------------------
def is_yellow(obj):   return obj["colour"] == "yellow"
def is_circle(obj):   return obj["shape"]  == "circle"

def xor_label(obj) -> bool:
    return is_yellow(obj) ^ is_circle(obj)          # Boolean xor

def decomp_vars(obj):
    """Return Z1,Z2,Z3,output for Model B."""
    y = is_yellow(obj); c = is_circle(obj)
    z1 =  y or c
    z2 =  y and c
    z3 =  not z2
    out=  z1 and z3
    return z1, z2, z3, out                         # all Booleans

# ---------------------------------------------------------------------
# 2. prompt template (fixed length – we pad to 64 tokens)
# ---------------------------------------------------------------------
alpaca_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Decide whether the object should be labelled True or False under the rule "yellow XOR circle".

### Input:
%s

### Response:
"""

def object_to_line(obj):
    return f"- {CO_ABR[obj['colour']]} {SH_ABR[obj['shape']]}"

# ---------------------------------------------------------------------
# 3. single (prompt, target) example  ---------------------------------
# ---------------------------------------------------------------------
def make_prompt(tokenizer, obj):
    prompt_txt = alpaca_template % object_to_line(obj)
    inp = tokenizer(prompt_txt, return_tensors="pt").input_ids[0]

    # output string and labels
    label_word = "True" if xor_label(obj) else "False"
    target_txt = f"{object_to_line(obj)} -> {label_word}"
    tgt = tokenizer(target_txt, return_tensors="pt").input_ids[0]

    # concatenate prompt + target just like price-tagging code
    input_ids = torch.cat([inp, tgt])
    labels    = input_ids.clone()
    labels[:len(inp)] = IGNORE          # only supervise the target part
    return input_ids, labels

# ---------------------------------------------------------------------
# 4. utilities for counterfactual pairs used by Boundless DAS
# ---------------------------------------------------------------------
def random_obj():
    return {"colour": random.choice(COLOURS),
            "shape" : random.choice(SHAPES)}

# ----------  Model A (single XOR variable) ---------------------------
def cf_pair_model_A(tokenizer):
    # choose two objects that differ in xor truth value
    while True:
        base = random_obj(); src = random_obj()
        if xor_label(base) != xor_label(src):
            break
    # We want the model, *after swap*, to output src’s label
    ctf_lab_word = "True" if xor_label(src) else "False"

    base_ids , _ = make_prompt(tokenizer, base)
    src_ids  , _ = make_prompt(tokenizer, src)

    # build counter-factual label vector (same length as base prompt)
    ctf_labels = torch.full_like(base_ids, IGNORE)
    ctf_labels[-1] = tokenizer.convert_tokens_to_ids(ctf_lab_word)

    return base_ids, src_ids, ctf_labels, 0         # 0 = single variable

# ----------  Model B (three decomposition variables) -----------------
def sample_for_B(var_id:int, desired_out:bool):
    """Pick base,src s.t. src differs from base *only* in var_id and
       src’s final xor == desired_out."""
    while True:
        base = random_obj(); src = random_obj()
        zb = decomp_vars(base); zs = decomp_vars(src)
        if zb[var_id] != zs[var_id] and zs[-1] == desired_out:
            return base, src

def cf_pair_model_B(tokenizer):
    var_id  = random.choice([0,1,2])           # which latent to swap
    wanted  = random.choice([False, True])     # desired final label
    base,src = sample_for_B(var_id, wanted)

    base_ids,_ = make_prompt(tokenizer, base)
    src_ids ,_ = make_prompt(tokenizer, src)

    ctf_labels = torch.full_like(base_ids, IGNORE)
    ctf_word   = "True" if wanted else "False"
    ctf_labels[-1] = tokenizer.convert_tokens_to_ids(ctf_word)
    return base_ids, src_ids, ctf_labels, var_id   # var_id = 0/1/2

# ---------------------------------------------------------------------
# 5. dataloader builders  ---------------------------------------------
# ---------------------------------------------------------------------
from datasets import Dataset
from torch.utils.data import DataLoader

def build_loader(tokenizer, generator_fn, n_examples:int,
                 batch_size:int=16, shuffle=True):
    bases, srcs, labels, ids = [],[],[],[]
    for _ in range(n_examples):
        b,s,l,i = generator_fn(tokenizer)
        bases.append(b.tolist())
        srcs .append(s.tolist())
        labels.append(l.tolist())
        ids   .append(i)
    ds = Dataset.from_dict({
        "input_ids"      : bases,
        "source_input_ids":srcs,
        "labels"         : labels,
        "intervention_ids":ids,
    }).with_format("torch")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)