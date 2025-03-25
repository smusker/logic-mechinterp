from utilities import compare_rules, assess_rule_consistency

# Install Pyvene if not already installed
"""
try:
    import pyvene
except ModuleNotFoundError:
    !pip install git+https://github.com/stanfordnlp/pyvene.git
"""

# Import necessary modules
import torch
from pyvene import (
    IntervenableModel,
    IntervenableConfig,
    RepresentationConfig,
    VanillaIntervention,
)
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pandas as pd

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ['TRANSFORMERS_CACHE'] = '/oscar/scratch/dkang33/model_cache'
model_id = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    #device_map="auto" #(removed as code expects model to be on one device),
    cache_dir=os.environ['TRANSFORMERS_CACHE']
)
model.to(device)
model.eval()

prompt = "The Space Needle is in downtown "

prompt = '''We will abbreviate sizes as: S=small, M=medium, L=large; colors as: BLU=blue, GRN=green, YEL=yellow, RED=red; shapes as: CIR=circle, TRI=triangle, REC=rectangle, SQR=square.

Learn the secret rule to label the objects correctly. If an object follows the rule, it should be labeled 'True'. Otherwise, it should be labeled 'False'. Always follow the arrow '->' with an object's label.

Here are some objects and their labels:
- Size=M Color=YEL Shape=TRI -> True
- Size=S Color=YEL Shape=CIR -> False
- Size=M Color=YEL Shape=REC -> True
- Size=M Color=BLU Shape=REC -> True
- Size=L Color=BLU Shape=CIR -> False
- Size=L Color=GRN Shape=REC -> True

Label the following object:
- Size=S Color=GRN Shape=CIR

Provide the label for the object above, then explain the rule you've inferred starting with "Based on your examples, I've learned that"

- Size=S Color=GRN Shape=CIR ->'''

prompt2 = '''We will abbreviate sizes as: S=small, M=medium, L=large; colors as: BLU=blue, GRN=green, YEL=yellow, RED=red; shapes as: CIR=circle, TRI=triangle, REC=rectangle, SQR=square.

Learn the secret rule to label the objects correctly. If an object follows the rule, it should be labeled 'True'. Otherwise, it should be labeled 'False'. Always follow the arrow '->' with an object's label.

Here are some objects and their labels:
- Size=L Color=BLU Shape=CIR -> True
- Size=L Color=YEL Shape=TRI -> False
- Size=L Color=BLU Shape=SQR -> True
- Size=S Color=BLU Shape=CIR -> True
- Size=S Color=GRN Shape=CIR -> False
- Size=S Color=RED Shape=REC -> True

Label the following object:
- Size=S Color=GRN Shape=SQR

Provide the label for the object above, then explain the rule you've inferred starting with 'Based on your examples, I've learned that'

'''

# inputs = tokenizer(prompt, return_tensors='pt').to(device)

# print(inputs)
# print(inputs['input_ids'])

# with torch.no_grad():
#     output_sequences = model.generate(
#         input_ids=inputs['input_ids'],
#         max_new_tokens=50,
#         do_sample=False,
#         temperature=0.0,  # Deterministic output
#         pad_token_id=tokenizer.pad_token_id,
#     )

# output_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

# print(output_text)

print(len(prompt))
print(len(prompt2))