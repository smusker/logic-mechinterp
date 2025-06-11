from transformers import AutoTokenizer

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

CO_ABR = {"yellow": "ye", "blue": "bl", "green": "gre", "red": "red"}
SH_ABR = {"circle": "cir", "triangle": "tri", "square": "squ"}

def bad_tokens(vocab):
    offenders = []
    for name, abr in vocab.items():
        ids = tokenizer(abr, add_special_tokens=False).input_ids
        if len(ids) != 1:
            offenders.append((name, abr, len(ids)))
    return offenders

bad = bad_tokens(CO_ABR) + bad_tokens(SH_ABR)

if not bad:
    print("✅ All abbreviations are single tokens.")
else:
    print("❌ The following abbreviations are NOT single tokens:")
    for name, abr, n in bad:
        print(f"  {name:9s}  '{abr}'  → {n} tokens")


#for checking prompt length

"""

from transformers import AutoTokenizer
from xor_utils   import make_prompt, random_obj

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

dummy_ids, _ = make_prompt(tokenizer, random_obj())
print("sequence length =", dummy_ids.size(0))

"""



"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto")
t = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
out = m(**t("hello", return_tensors="pt").to(next(m.parameters()).device))
print("✔ multi-GPU forward works")
"""