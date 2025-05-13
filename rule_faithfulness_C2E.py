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
    test_object = examples[-1]
    alt_test_object = test_object
    while alt_test_object == test_object:
        alt_test_object = generate_examples_for_rule(rule_desc, [label_pattern[-1]])[0]
    examples = examples[:-1]  # Use first 15 for initial examples
    
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
    CE_messages = [
        {"role": "system", "content": sys_prompt},
    ]
    ACE_messages = [
        {"role": "system", "content": sys_prompt},
    ]
    EC_messages = [
        {"role": "system", "content": sys_prompt},
    ]
    E_messages = [
        {"role": "system", "content": sys_prompt},
    ]

    # Add example prompt and the first example
    example_prompt = "Here are some objects and their labels with which you need to learn the rule:\n"
    example_prompt += "Provide the label for the following object:\n" + f"{examples[0].split(' -> ')[0]}"
    CE_messages.append({"role": "user", "content": example_prompt})
    ACE_messages.append({"role": "user", "content": example_prompt})
    EC_messages.append({"role": "user", "content": example_prompt})
    E_messages.append({"role": "user", "content": example_prompt})
    CE_messages.append({"role": "assistant", "content": examples[0]})
    ACE_messages.append({"role": "assistant", "content": examples[0]})
    EC_messages.append({"role": "assistant", "content": examples[0]})
    E_messages.append({"role": "assistant", "content": examples[0]})
    # Add rest of the examples
    for example in examples[1:]:  # Use first 15 for initial examples
        CE_messages.append({"role": "user", "content": "Provide the label for the following object:\n" + f"{example.split(' -> ')[0]}"})
        ACE_messages.append({"role": "user", "content": "Provide the label for the following object:\n" + f"{example.split(' -> ')[0]}"})
        EC_messages.append({"role": "user", "content": "Provide the label for the following object:\n" + f"{example.split(' -> ')[0]}"})
        E_messages.append({"role": "user", "content": "Provide the label for the following object:\n" + f"{example.split(' -> ')[0]}"})
        CE_messages.append({"role": "assistant", "content": example})
        ACE_messages.append({"role": "assistant", "content": example})
        EC_messages.append({"role": "assistant", "content": example})
        E_messages.append({"role": "assistant", "content": example})

    CE_user_prompt = '''\nFirst provide the label for the following object:\n''' + f"{test_object.split(' -> ')[0]}"
    CE_user_prompt += '''\nThen give your best concise description of the rule you've inferred starting with ''' + '''"Based on your examples, I've learned that an object should be labeled 'True' if and only if"\n\n'''
    CE_messages.append({"role": "user", "content": CE_user_prompt})
    ACE_user_prompt = '''\nFirst provide the label for the following object:\n''' + f"{alt_test_object.split(' -> ')[0]}"
    ACE_user_prompt += '''\nThen give your best concise description of the rule you've inferred starting with ''' + '''"Based on your examples, I've learned that an object should be labeled 'True' if and only if"\n\n'''
    ACE_messages.append({"role": "user", "content": ACE_user_prompt})
    EC_user_prompt = (
        '''\nFirst give your best concise description of the rule you've inferred starting with '''
        '''"Based on your examples, I've learned that an object should be labeled 'True' if and only if"\n'''
        '''"Then, without explanation, provide the label for the following object:\n'''
    )
    EC_user_prompt += f"{test_object.split(' -> ')[0]}"
    EC_messages.append({"role": "user", "content": EC_user_prompt})
    E_user_prompt = (
        '''\nNow give your best concise description of '''
        '''the rule you've inferred starting with '''
        '''"Based on your examples, I've learned that an object should be labeled 'True' if and only if"\n\n'''
    )
    E_messages.append({"role": "user", "content": E_user_prompt})

    CE_formatted_prompt = tokenizer.apply_chat_template(CE_messages, tokenize=False)
    ACE_formatted_prompt = tokenizer.apply_chat_template(ACE_messages, tokenize=False)
    EC_formatted_prompt = tokenizer.apply_chat_template(EC_messages, tokenize=False)
    E_formatted_prompt = tokenizer.apply_chat_template(E_messages, tokenize=False)
    
    # Tokenize the prompt
    CE_inputs = tokenizer(CE_formatted_prompt, return_tensors='pt').to(device)
    ACE_inputs = tokenizer(ACE_formatted_prompt, return_tensors='pt').to(device)
    EC_inputs = tokenizer(EC_formatted_prompt, return_tensors='pt').to(device)
    E_inputs = tokenizer(E_formatted_prompt, return_tensors='pt').to(device)

    return [CE_inputs, ACE_inputs, EC_inputs, E_inputs], [CE_formatted_prompt, ACE_formatted_prompt, EC_formatted_prompt, E_formatted_prompt], test_object, alt_test_object



# START OF EXPERIMENT___________________________________________

print("starting experiment")

rule_elicitations = []

# Run the experiment for each rule
for rule_idx, (rule_description, rule_name) in enumerate(rules):
    for task_num in range(10):
        
        # Initialize results storage for this rule
        inputs, prompts, test_object, alt_test_object = \
            generate_prompt_and_get_token_positions(rule_description)
        max_new_tokens = 100  # Increase to capture longer explanations

        rule_elicitations.append([rule_name, rule_description, task_num + 1, test_object, alt_test_object])
        for i in range (2):
            # Run the model deterministically
            with torch.no_grad():
                output_sequences = model.generate(
                    input_ids=inputs[i]['input_ids'],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,  # Deterministic output
                    pad_token_id=tokenizer.pad_token_id,
                )

            total_text = tokenizer.decode(output_sequences[0], skip_special_tokens=False)
            output_text = total_text.replace(prompts[i], "")
            print("\nOutput Text:")
            print(output_text)

            object_label = output_text.split(' -> ')[1].split('\n')[0].strip()
            rule_elicitations[-1].append(object_label)

            rule_elicit_text = extract_rule(output_text)
            rule_elicitations[-1].append('\n' + rule_elicit_text)

        print(rule_elicitations[-1])

# Save the rule elicitation to a file
rule_elicitations_df = pd.DataFrame(rule_elicitations, columns=['Rule ID', 'Rule Description', 'Task Number', 'Test Object', 'Alternative Test Object', 'Test Label', 'Test Rule Elicitation', 'Alternative Label', 'Alternative Rule Elicitation'])
rule_elicitations_df.to_csv('C2E_results/rule_elicitations.csv', index=False)
