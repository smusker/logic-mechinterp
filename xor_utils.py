# --------------------------------------------------------------------------------
# xor_utils.py  •  infer‑the‑rule ▶︎ label‑the‑object  (single‑token safe)
# --------------------------------------------------------------------------------
#  • Each prompt shows 12 worked examples; model infers the XOR rule then
#    labels one final object.
#  • Abbreviations are chosen so *each one* is a single tokenizer piece for
#    Meta‑Llama‑3‑8B‑Instruct, guaranteeing constant prompt length without pad.
# --------------------------------------------------------------------------------

import random, torch
from typing import List, Optional
from datasets import Dataset
from torch.utils.data import DataLoader

IGNORE = -100  # label padding

# ---------------------------------------------------------------------
# 1. Vocabulary & single‑token abbreviations  (verified with a quick script)
# ---------------------------------------------------------------------
COLOURS = ["yellow", "blue", "green", "red"]
SHAPES  = ["circle", "triangle", "square"]

CO_ABR = {"yellow": "ye",  # ▁ye
          "blue"  : "bl",  # ▁bl
          "green" : "gre", # ▁gre
          "red"   : "red"} # ▁red

SH_ABR = {"circle"   : "cir",  # ▁cir
          "triangle" : "tri",  # ▁tri
          "square"   : "squ"}  # ▁squ

def label_id(tok, truth: bool) -> int:
    txt = "True" if truth else "False"     # ← no leading space
    return tok.encode(txt, add_special_tokens=False)[0]



# ---------------------------------------------------------------------
# 2. XOR rule helpers
# ---------------------------------------------------------------------

def is_yellow(o): return o["colour"] == "yellow"

def is_circle(o): return o["shape"]  == "circle"

def xor_label(o): return is_yellow(o) ^ is_circle(o)

# decomposition for Model‑B hypothesis

def decomp_vars(o):
    y, c = is_yellow(o), is_circle(o)
    z1 = y or c; z2 = y and c; z3 = not z2
    return z1, z2, z3, (z1 and z3)

# ---------------------------------------------------------------------
# 3. Prompt template
# ---------------------------------------------------------------------
ALPACA_TEMPLATE = (
    """\n### Instruction:\nYou will see several examples. Infer the labelling rule, then output True or False for the final object.\n\n### Input:\n%s\n\n### Response:\n"""
)

# ---------------------------------------------------------------------
# 4. Line builder
# ---------------------------------------------------------------------

def object_to_line(obj, label: Optional[str] = None):
    core = f"- {CO_ABR[obj['colour']]} {SH_ABR[obj['shape']]}"
    return core if label is None else f"{core} -> {label}"

# ---------------------------------------------------------------------
# 5. Sampling utilities
# ---------------------------------------------------------------------

def random_obj():
    return {"colour": random.choice(COLOURS), "shape": random.choice(SHAPES)}


def sample_examples(n=10, *, exclude=None):#must be less than or equal to 10 otherwise not enough objects and will be infinite loop
    """
    Return `n` distinct (colour, shape) objects that are **not** in `exclude`.
    `exclude` can be a single dict or an iterable of dicts.
    """
    # 1 ─ build the forbidden-set  (O(1) lookup later)
    if exclude is None:
        forbidden = set()
    elif isinstance(exclude, dict):
        forbidden = {(exclude["colour"], exclude["shape"])}
    else:
        forbidden = {(o["colour"], o["shape"]) for o in exclude}

    # 2 ─ safety check: do we have enough unique objects?
    MAX_UNIQUE = len(COLOURS) * len(SHAPES)          # 4 × 3 = 12
    assert n <= MAX_UNIQUE - len(forbidden), (
        f"Requested {n} unique examples but only "
        f"{MAX_UNIQUE - len(forbidden)} are available."
    )

    # 3 ─ sample without replacement
    examples, seen = [], set()
    while len(examples) < n:
        o = random_obj()
        key = (o["colour"], o["shape"])
        if key in forbidden or key in seen:
            continue
        examples.append(o)
        seen.add(key)
    return examples

# ---------------------------------------------------------------------
# 6. Prompt builder (returns token ids + label mask)
# ---------------------------------------------------------------------

def make_prompt(tok, examples: List[dict], query_obj: dict):
    ex_lines = [object_to_line(o, "True" if xor_label(o) else "False") for o in examples]
    query_line = object_to_line(query_obj) + " ->"
    prompt_body = "\n".join(ex_lines + ["", "Now label the final object:", query_line])
    prompt_txt  = ALPACA_TEMPLATE % prompt_body
    full_txt = prompt_txt + ("True" if xor_label(query_obj) else "False")

    ids  = tok(full_txt, return_tensors="pt").input_ids[0]
    lbls = ids.clone(); lbls[:-1] = IGNORE
    return ids, lbls

# ---------------------------------------------------------------------
# 7. Counter‑factual pair generators
# ---------------------------------------------------------------------

def cf_pair_model_A(tok):
    while True:
        base, src = random_obj(), random_obj()
        if xor_label(base) != xor_label(src):
            break
    ex = sample_examples(exclude=[base, src])
    b_ids, _ = make_prompt(tok, ex, base)
    s_ids, _ = make_prompt(tok, ex, src)

    ctf = torch.full_like(b_ids, IGNORE)
    ctf[-1] = label_id(tok, xor_label(src))
    return b_ids, s_ids, ctf, 0


def sample_for_B(var_id: int, desired_out: bool):
    while True:
        b, s = random_obj(), random_obj()
        if decomp_vars(b)[var_id] != decomp_vars(s)[var_id] and decomp_vars(s)[-1] == desired_out:
            return b, s


def cf_pair_model_B(tok):
    var_id  = random.choice([0, 1, 2])
    desired = random.choice([False, True])
    base, src = sample_for_B(var_id, desired)
    ex = sample_examples(exclude=[base, src])
    b_ids, _ = make_prompt(tok, ex, base)
    s_ids, _ = make_prompt(tok, ex, src)

    ctf = torch.full_like(b_ids, IGNORE)
    ctf[-1] = label_id(tok, desired)
    return b_ids, s_ids, ctf, var_id

# ---------------------------------------------------------------------
# 8. DataLoader builder
# ---------------------------------------------------------------------

def build_loader(tok, generator_fn, n_examples: int, batch_size=16, shuffle=True):
    recs = [generator_fn(tok) for _ in range(n_examples)]
    bases, srcs, labels, ids = zip(*[(b.tolist(), s.tolist(), l.tolist(), i) for b, s, l, i in recs])
    ds = Dataset.from_dict({
        "input_ids": bases,
        "source_input_ids": srcs,
        "labels": labels,
        "intervention_ids": ids,
    }).with_format("torch")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
