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
    ConstantSourceIntervention,
    LocalistRepresentationIntervention
)
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pandas as pd
import copy
import csv
from dotenv import load_dotenv
from datasets import load_dataset
import json

load_dotenv()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load LLaMA 3.1 8B using Auto classes
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

ds = load_dataset("akariasai/PopQA", split='test').filter(lambda example: len(example["s_aliases"].split(',')) >= 2)
print(ds)
print(ds[0])

def query_llama(question):
    inputs = tokenizer(
        question,
        return_tensors='pt').to(device)
    with torch.no_grad():
        output = model(
            input_ids=inputs['input_ids'],
            do_sample=False,
            temperature=0.0,  # Deterministic output
            pad_token_id=tokenizer.pad_token_id,
        )
    output_token = torch.argmax(torch.nn.functional.softmax(output.logits, dim = -1)[0][-1])
    output_token_string = tokenizer.decode(output_token, skip_special_tokens=False)
    return output_token_string.strip().lower()

# 1. successfully answers the question, 2. if all aliases successfully answer the question, 

results = []
for i in range(100):
    result = []
    example = ds[i]
    subject = example["subj"]
    property = example["prop"]
    answers = json.loads(example["possible_answers"])
    print(f"{subject}'s {property} are {answers}")
    aliases = json.loads(example["s_aliases"])
    if len(aliases) < 2:
        print(f"Example {i} has less than 2 aliases: {example}")

    header = '''Hans Zimmer's occupation is: composer
    France's capital is: Paris
    Space Odyssey's director is: Stanley Kubrick
    Richard Feynman's occupation is: physicist
    Saudi Arabia's religion is: Islam
    Saw 2's genre is: horror
    The Godfather's director is: Francis Ford Coppola
    Teddy Roosevelt's father is: Theodore Roosevelt Sr.
    '''

    question = f"{subject}'s {property} is:"
    subj_answer = query_llama(header + question)
    print(f"{subject}'s {property} is {subj_answer}")

    if subj_answer in answers:
        result.append(1)
    else:
        result.append(0)

    alias_answers = 1
    alias_consistency = 1
    for alias in aliases:
        question = f"{alias}'s {property} is:"
        alias_answer = query_llama(header + question)

        if alias_answer in answers:
            alias_answers = alias_answers * 1
        else:
            alias_answers = alias_answers * 0

        if alias_answer == subj_answer:
            alias_consistency = alias_consistency * 1
        else:
            alias_consistency = alias_consistency * 0

        print(f"{alias}'s {property} is: {alias_answer}")

    result.append(alias_answers)
    result.append(alias_consistency)
    results.append(result)

correct_rows = [x for x in results if x[0] == 1]
print(f"Number of examples where the subject's answer is correct: {len(correct_rows)}")
print(f"Ratio of examples where all aliases are consistent with the subject's answer: {sum(x[2] for x in results)/len(results)}")
print(f"Number of examples where all aliases' answers were also correct when the subject's answer was correct: {sum(x[1] for x in correct_rows)}")
print(f"Ratio of correct alias generalization: {sum(x[1] for x in correct_rows)/len(correct_rows)}")

# 41, 0.3, 22, 0.5365853658536586

with open("output.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(results)