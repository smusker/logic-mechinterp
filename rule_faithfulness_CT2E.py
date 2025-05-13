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
import copy

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load LLaMA 3.1 using Auto classes
os.environ['TRANSFORMERS_CACHE'] = '/oscar/scratch/dkang33/model_cache'
model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    #device_map="auto" #(removed as code expects model to be on one device),
    cache_dir=os.environ['TRANSFORMERS_CACHE']
)
model.to(device)
model.eval()

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Create directories to save heatmaps, logs, and intervention records
os.makedirs('C2E_results', exist_ok=True)

# Abbreviations for sizes, colors, and shapes
sizes = ['S', 'M', 'L']  # Small, Medium, Large
colors = ['BLU', 'GRN', 'YEL']  # Blue, Green, Yellow
shapes = ['CIR', 'TRI', 'REC']  # Circle, Triangle, Rectangle

# Define the rules and their labels
rules = [
    ("An object is labeled 'True' if it is not a CIR.", "hg05"),
    ("An object is labeled 'True' if it is BLU or a CIR.", "hg06"),
    ("An object is labeled 'True' if it is a CIR or a TRI.", "hg07"),
    ("An object is labeled 'True' if it is a CIR and not BLU.", "hg10"),
    ("An object is labeled 'True' if it is S and BLU.", "hg24"),
]

def label_object(description, rule_desc):
    tokens = description.replace('-', '').split()
    attributes = dict(token.split('=') for token in tokens)
    size = attributes.get('Size')
    color = attributes.get('Color')
    shape = attributes.get('Shape')

    if "not a CIR" in rule_desc:
        is_cir = 'CIR' == shape
        return 'False' if is_cir else 'True'
    elif "BLU or a CIR" in rule_desc:
        is_blu = 'BLU' == color
        is_cir = 'CIR' == shape
        return 'True' if is_blu or is_cir else 'False'
    elif "a CIR or a TRI" in rule_desc:
        is_cir = 'CIR' == shape
        is_tri = 'TRI' == shape
        return 'True' if is_cir or is_tri else 'False'
    elif "CIR and not BLU" in rule_desc:
        is_cir = 'CIR' == shape
        is_blu = 'BLU' == color
        return 'True' if is_cir and not is_blu else 'False'
    elif "S and BLU" in rule_desc:
        is_s = 'S' == size
        is_blu = 'BLU' == color
        return 'True' if is_s and is_blu else 'False'
    else:
        return 'False'  # Default fallback

def generate_random_object():
    size = random.choice(sizes)
    color = random.choice(colors)
    shape = random.choice(shapes)
    obj = f"- Size={size} Color={color} Shape={shape}"
    return obj

def generate_true_example(rule_desc):
    while True:
        obj = generate_random_object()
        if label_object(obj, rule_desc) == 'True':
            return obj

def generate_false_example(rule_desc):
    while True:
        obj = generate_random_object()
        if label_object(obj, rule_desc) == 'False':
            return obj

def generate_examples_for_rule(rule_desc, label_pattern):
    examples = []
    for label in label_pattern:
        if label == 'True':
            example = generate_true_example(rule_desc)
        else:
            example = generate_false_example(rule_desc)
        examples.append(f"{example} -> {label}")
    return examples

def extract_rule(output_text):
    start_phrase = "Based on your examples, I've learned that an object should be labeled 'True' if and only if"
    start_idx = output_text.find(start_phrase)
    if start_idx != -1:
        output_text = output_text[start_idx:].replace(start_phrase, "")
        output_text = output_text[:output_text.find('\n\n')]
        return output_text
    else:
        print("Rule start phrase not found in the output text.")
        return ""

def generate_prompt_and_get_token_positions(rule_desc):
    # Define the label pattern
    labels = ['True', 'False']
    label_pattern = random.choices(labels, k=16)
    # Generate examples according to the label pattern
    examples = generate_examples_for_rule(rule_desc, label_pattern)
    test_object_line = examples[-1]
    examples = examples[:-1]  # Remove the last example for the test object
    
    # Build the prompt with abbreviations up front
    abbreviation_section = (
        "We will abbreviate sizes as: S=small, M=medium, L=large; "
        "colors as: BLU=blue, GRN=green, YEL=yellow; "
        "shapes as: CIR=circle, TRI=triangle, REC=rectangle.\n\n"
    )
    instructions = (
        "Learn the secret rule that labels the objects into 'True' or 'False'. "
        "If an object follows the rule, it is labeled 'True'. "
        "Otherwise, it is labeled 'False'. "
        "Always follow the arrow '->' to find the object's label.\n\n"
    )
    # Combine instructions and abbreviations
    sys_prompt = abbreviation_section + instructions
    messages = [
        {"role": "system", "content": sys_prompt},
    ]

    # Add example prompt and the first example
    example_prompt = "Here are some objects and their labels with which you need to learn the rule:\n"
    example_prompt += "Provide the label for the following object:\n" + f"{examples[0].split(' -> ')[0]}"
    messages.append({"role": "user", "content": example_prompt})
    messages.append({"role": "assistant", "content": examples[0]})
    # Add rest of the examples
    for example in examples[1:]:  # Use first 15 for initial examples
        messages.append({"role": "user", "content": "Provide the label for the following object:\n" + f"{example.split(' -> ')[0]}"})
        messages.append({"role": "assistant", "content": example})

    user_prompt = '''\nFirst, without explanation, provide the label for the following object:\n''' + f"{test_object_line.split(' -> ')[0]}"
    user_prompt += '''\nThen give your best concise description of the rule you've inferred starting with ''' + '''"Based on your examples, I've learned that an object should be labeled 'True' if and only if"\n\n'''
    messages.append({"role": "user", "content": user_prompt})

    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # Tokenize the prompt
    inputs = tokenizer(formatted_prompt, return_tensors='pt').to(device)
    
    return formatted_prompt, inputs, test_object_line




# START OF EXPERIMENT___________________________________________

rule_elicitations = []

print("starting experiment")

# Run the experiment for each rule
for rule_idx, (rule_description, rule_name) in enumerate(rules):
    for task_num in range(10):
    
        # Initialize results storage for this rule
        prompt, inputs, test_object_line = \
            generate_prompt_and_get_token_positions(rule_description)
        max_new_tokens = 100  # Increase to capture longer explanations

        print(prompt)

        # Run the model deterministically
        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=inputs['input_ids'],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,  # Deterministic output
                pad_token_id=tokenizer.pad_token_id,
            )

        total_text = tokenizer.decode(output_sequences[0], skip_special_tokens=False)
        # print('Clean total text:\n')
        # print(total_text)
        output_text = total_text.replace(prompt, '')
        print('Clean output text:\n')
        print(output_text)
        object_label = output_text.split(' -> ')[1].split('\n')[0].strip()
        print('Clean object label:')
        print(object_label)
        if object_label != test_object_line.split(' -> ')[1].strip():
            print("Task object label is not the true label")
        
        elicit_rule = '\n' + extract_rule(output_text)
        print('Clean elicited rule:\n')
        print(elicit_rule)

        alt_label = 'True' if object_label == 'False' else 'False'
        corrupt_text = total_text[:total_text.rfind('->')] + '-> ' + alt_label
        corrupt_input = tokenizer(corrupt_text, return_tensors='pt').to(device)
        corrupt_output_sequences = model.generate(
            input_ids=corrupt_input['input_ids'],  
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,  # Deterministic output
            pad_token_id=tokenizer.pad_token_id,
        )
        corrupt_total_text = tokenizer.decode(corrupt_output_sequences[0], skip_special_tokens=False)
        # print('Corrupt total text:\n')
        # print(corrupt_total_text)
        corrupt_output_text = corrupt_total_text.replace(prompt, '')
        print('Corrupt output text:\n')
        print(corrupt_output_text)
        corrupt_elicit_rule = '\n' + extract_rule(corrupt_output_text) + '\n\n'
        print('Corrupt elicited rule:\n')
        print(corrupt_elicit_rule)

        rule_elicitations.append([rule_name, rule_description, task_num, test_object_line, elicit_rule, corrupt_elicit_rule])

rule_elicitations_df = pd.DataFrame(rule_elicitations, columns=['Rule ID', 'Rule Description', 'Task Number', 'Test Object', 'Clean Rule', 'Corrupt Rule'])
rule_elicitations_df.to_csv('C2E_results/CT2E_rule_elicitations.csv', index=False)

        