from transformers import AutoTokenizer
from xor_utils   import make_prompt, random_obj

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

dummy_ids, _ = make_prompt(tokenizer, random_obj())
print("sequence length =", dummy_ids.size(0))

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