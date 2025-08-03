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

load_dotenv()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load LLaMA 3.1 8B using Auto classes
os.environ['TRANSFORMERS_CACHE'] = '/oscar/scratch/dkang33/model_cache'
model_id = "meta-llama/Llama-3.3-70B-Instruct"
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
os.makedirs('heatmaps', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('intervention_records', exist_ok=True)

# Abbreviations for sizes, colors, and shapes
sizes = ['S', 'M', 'L']  # Small, Medium, Large
colors = ['BLU', 'GRN', 'YEL']  # Blue, Green, Yellow
shapes = ['CIR', 'TRI', 'REC']  # Circle, Triangle, Rectangle

# Define the rules and their labels
rules = [
    # ("An object is labeled 'True' if it is not a CIR.", "hg05"),
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

def label_test_object_in_output(output_text, test_object_line):
    """
    Extracts the label assigned to the test object in the model's output.
    """
    try:
        # Find the line that contains the test object
        lines = output_text.split('\n')
        for i, line in enumerate(lines):
            if test_object_line.strip() in line:
                # The label is expected to be in the same line or next line
                if '->' in line:
                    parts = line.split('->')
                    if len(parts) == 2:
                        assigned_label = parts[1].strip().split()[0]
                        if assigned_label in ['True', 'False']:
                            return assigned_label
                else:
                    # Check the next line
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line in ['True', 'False']:
                            return next_line
            # In case the model rephrases the test object line
            elif 'Shape=' in line and 'Color=' in line and 'Size=' in line:
                parts = line.split('->')
                if len(parts) == 2:
                    assigned_label = parts[1].strip().split()[0]
                    if assigned_label in ['True', 'False']:
                        return assigned_label
        return None  # Label not found
    except Exception:
        return None

def generate_prompt_and_get_token_positions(rule_desc):
    # Define the label pattern
    labels = ['True', 'False']
    label_pattern = random.choices(labels, k=16)
    # Generate examples according to the label pattern
    examples = generate_examples_for_rule(rule_desc, label_pattern)
    test_object_line = examples[-1].split(' -> ')[0]
    examples = examples[:-1]
    
    # Build the prompt with abbreviations up front
    abbreviation_section = (
        "We will abbreviate sizes as: S=small, M=medium, L=large; "
        "colors as: BLU=blue, GRN=green, YEL=yellow; "
        "shapes as: CIR=circle, TRI=triangle, REC=rectangle.\n\n"
    )
    instructions = (
        "Learn the secret rule to label the objects correctly. "
        "If an object follows the rule, it should be labeled 'True'. "
        "Otherwise, it should be labeled 'False'. "
        "Always follow the arrow '->' with an object's label.\n\n"
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

    intro_prompt = tokenizer.apply_chat_template(messages, tokenize=False)

    task_prompt = "\nFirst, provide the label for the following object:\n"
    task_prompt += f"{test_object_line}\n"
    task_prompt += (
        '''then also explain '''
        '''the rule you've inferred starting with '''
        '''"Based on your examples, I've learned that an object should be labeled 'True' if and only if"\n\n'''
    )

    messages.append({"role": "user", "content": task_prompt})
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # Tokenize the prompt
    inputs = tokenizer(formatted_prompt, return_tensors='pt').to(device)
    tokenized_prompt = tokenizer.convert_ids_to_tokens(
        inputs['input_ids'][0]
    )
    
    # Identify the start index for interventions (after intro tokens)
    intro_inputs = tokenizer(intro_prompt, return_tensors='pt').to(device)
    intro_token_length = intro_inputs['input_ids'].shape[1]
    
    return formatted_prompt, inputs, tokenized_prompt, test_object_line, intro_token_length

def get_activation_capturer(activations_dict, key):
    def capturer(module, input, output):
        if key not in activations_dict:
            activations_dict[key] = []
        # Check if output is a tuple
        if isinstance(output, tuple):
            # Extract the tensor you want to capture
            output_to_save = output[0]
        else:
            output_to_save = output
        # Detach and save the output
        activations_dict[key].append(output_to_save.detach())
    return capturer

def register_hooks(activations, layer_nums, components_to_capture):
    handles = []
    for layer_num in layer_nums:
        layer = model.model.layers[layer_num]  # Adjust reference for LLaMA
        for component in components_to_capture:
            key = (component, layer_num)
            # Set up the hooks based on LLaMA architecture
            if component == 'block_output':
                handle = layer.register_forward_hook(
                    get_activation_capturer(activations, key)
                )
            elif component == 'mlp_output':
                handle = layer.mlp.register_forward_hook(
                    get_activation_capturer(activations, key)
                )
            elif component == 'attention_output':
                handle = layer.self_attn.register_forward_hook(  # Adjust for LLaMA
                    get_activation_capturer(activations, key)
                )
            handles.append(handle)
    return handles


def find_label_token_positions(inputs, label_to_noise='True'):
    """
    Find positions of 'True' or 'False' tokens in the tokenized prompt.
    """
    label_token_ids = tokenizer.convert_tokens_to_ids(['True', 'ĠTrue', 'False', 'ĠFalse'])
    positions = []
    input_ids = inputs['input_ids'][0]
    for idx, token_id in enumerate(input_ids):
        if token_id in label_token_ids:
            positions.append(idx)
    return positions

def add_noise_to_token_embeddings(inputs, token_positions, noise_level=0.1):
    """
    Add Gaussian noise to the embeddings of specific tokens.
    """
    noised_inputs = copy.deepcopy(inputs)
    with torch.no_grad():
        embedding_layer = model.model.embed_tokens  # Update for LLaMA
        # Get the original embeddings
        input_embeddings = embedding_layer(noised_inputs['input_ids'])
        # Generate noise
        noise = torch.randn_like(input_embeddings) * noise_level
        # Add noise to specific token positions
        for pos in token_positions:
            input_embeddings[0, pos, :] += noise[0, pos, :]
        # Replace the input_ids with embeddings
        noised_inputs['inputs_embeds'] = input_embeddings
        noised_inputs.pop('input_ids', None)
    return noised_inputs

class NoiseIntervention(ConstantSourceIntervention, LocalistRepresentationIntervention):
    def __init__(self, embed_dim, **kwargs):
        super().__init__()
        self.interchange_dim = embed_dim
        rs = np.random.RandomState(1)
        prng = lambda *shape: rs.randn(*shape)
        self.noise = torch.from_numpy(
            prng(1, 1, embed_dim)).to(device)
        self.noise_level = 0.13462981581687927

    def forward(self, base, source=None, subspaces=None):
        base[..., : self.interchange_dim] += self.noise * self.noise_level
        return base

    def __str__(self):
        return f"NoiseIntervention(embed_dim={self.embed_dim})"



# START OF EXPERIMENT___________________________________________

# Total number of layers in the model
num_layers = model.config.num_hidden_layers

# Components to intervene on
components = ['block_output', 'mlp_output', 'attention_output']# 

# Sliding window parameters
token_window_size = 2
token_step_size = 8
layer_window_size = 2
layer_step_size = 8

print("starting experiment")

# Run the experiment for each rule
for rule_idx, (rule_description, rule_name) in enumerate(rules):
    # Open a log file to write outputs
    log_filename = f"logs/{rule_name}_log.txt"
    log_file = open(log_filename, 'w')

    log_file.write(f"\n--- Processing {rule_name}: {rule_description} ---\n\n")
    
    # Initialize results storage for this rule
    prompt, inputs, tokenized_prompt, test_object_line, intro_token_length = \
        generate_prompt_and_get_token_positions(rule_description)
    max_new_tokens = 100  # Increase to capture longer explanations

    print(prompt)

    # Initialize variables to store runs
    base_run = None
    noised_run = None

    # Run the model deterministically
    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=inputs['input_ids'],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,  # Deterministic output
            pad_token_id=tokenizer.pad_token_id,
        )

    base_total_text = tokenizer.decode(output_sequences[0], skip_special_tokens=False)
    base_output_text = base_total_text.replace(prompt, "")
    base_label = label_test_object_in_output(base_output_text, test_object_line)
    base_rule = extract_rule(base_output_text)
    true_label = label_object(test_object_line, rule_description)

    log_file.write("Base Total Text: " + base_total_text + "\n")
    log_file.write("True Label:\n" + true_label + "\n")

    # Save the base run
    if base_label is not None:
        base_run = {
            'text': base_output_text,
            'assigned_label': base_label,
            'rule': base_rule,
            'tokens': tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]),
            'position_names': [f'Token_{i}' for i in range(inputs['input_ids'].shape[1])],
        }

    # Find positions of the label tokens in the prompt
    label_token_positions = find_label_token_positions(inputs)
    label_token_positions.pop(0)
    label_token_positions.pop(0)

    # Attempt to flip the output by adding noise
    flip_obtained = False
    noise_level = 0.3  # Adjust as needed
    
    # Initialize variables for noise intervention
    noise_pos_count = 0
    noise_pos_list = []
    noised_inputs = {}
    noised_output_text = ""
    noised_label = ""
    noised_rule = ""
    noise_config = IntervenableConfig(
        representations=[
            RepresentationConfig(
                0,              # layer
                "block_input",  # intervention type
            ),
        ],
        intervention_types=NoiseIntervention,
    )
    noise_intervenable = IntervenableModel(noise_config, model)

    # Add noise to the label token positions one at a time
    for pos in label_token_positions:
        noise_pos_count += 1
        noise_pos_list.append(pos)
        # Create a new input with noise added at position 'pos'
        noised_inputs = add_noise_to_token_embeddings(inputs, noise_pos_list, noise_level=noise_level)
        # Run the model with the noisy input
        # Generate output
        with torch.no_grad():
            noised_output_sequences = model.generate(
                inputs_embeds=noised_inputs['inputs_embeds'],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode the output
        noised_output_text = test_object_line + " ->" + tokenizer.decode(noised_output_sequences[0], skip_special_tokens=False)
        noised_label = label_test_object_in_output(noised_output_text, test_object_line)
        noised_rule = extract_rule(noised_output_text)

        print("\nNo. Noised: " + str(noise_pos_count))
        print("\nNoised Assigned Label:\n" + noised_label)
        print("\nNoised Elicited Rule:\n" + noised_rule)

        # Check if the categorization output has flipped
        if noised_label != base_label and noised_label is not None:
            flip_obtained = True
            break  # Exit the loop once the flip is obtained

    if not flip_obtained:
        print("\nnot flipped")
        # Optionally, try increasing noise level or adding noise to more tokens
        # log_file.write(f"Could not flip the output by adding noise to '{label_to_noise}' tokens.\n")
        log_file.write(f"Could not flip the output by adding noise to label tokens.\n")

        log_file.close()
        continue  # Skip to the next rule or handle this case accordingly
    else:
        print("\nflipped")
        # Save the runs
        noised_run = {
            'text': noised_output_text,
            'assigned_label': noised_label,
            'rule': noised_rule,
            'tokens': tokenizer.convert_ids_to_tokens(noised_inputs['inputs_embeds'].argmax(dim=-1)[0]),
            'position_names': [f'Token_{i}' for i in range(noised_inputs['inputs_embeds'].shape[1])],
        }

        log_file.write("\n\nBase Run\n")
        log_file.write("Text:" + base_run['text'] + "\n")
        log_file.write("Assigned Label:" + base_run['assigned_label'] + "\n")
        log_file.write("Rule:" + base_run['rule'] + "\n")

        log_file.write("\nNoised Run\n")
        log_file.write("Text:" + noised_run['text'] + "\n")
        log_file.write("Assigned Label:" + noised_run['assigned_label'] + "\n")
        log_file.write("Rule:" + noised_run['rule'] + "\n")

    # Proceed if both runs are collected
    if base_run is not None and noised_run is not None:
        # Use the token positions and names from the correct run
        num_positions = len(base_run['position_names'])

        # Prepare sliding windows for tokens
        token_windows = []
        for start in range(intro_token_length-16, num_positions, token_step_size):
            end = min(start + token_window_size, num_positions)
            token_windows.append((start, end))

        # Prepare sliding windows for layers
        layer_windows = []
        for start in range(0, num_layers, layer_step_size):
            end = min(start + layer_window_size, num_layers)
            layer_windows.append((start, end))

        # Initialize counts
        co_flip_counts = np.zeros((len(components), len(layer_windows), len(token_windows)))
        categorization_flip_counts = np.zeros((len(components), len(layer_windows), len(token_windows)))
        consistency_counts = np.zeros((len(components), len(layer_windows), len(token_windows)))

        # Initialize list for intervention records
        intervention_records = []
        intervention_records_fieldnames = ['intervention_component', 'intervention_layer_window', 'intervention_token_window',
                                 'rule_changed', 'categorization_consistent', 
                                 'base_label', 'base_rule',
                                 'noised_label', 'noised_rule',
                                 'restored_label', 'restored_rule']
        # Write to CSV
        with open(f"intervention_records/{rule_name}_interventions.csv", 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=intervention_records_fieldnames)
            writer.writeheader()
            csvfile.flush()

        # Counters for aggregate statistics
        total_cat_flips = 0
        total_co_flips = 0
        total_consistencies = 0

        # Ensure tokens and positions align
        if base_run['position_names'] != noised_run['position_names']:
            log_file.write("Token positions do not align between correct and incorrect runs.\n")
            log_file.close()
            continue  # Skip to next rule or handle accordingly

        # Perform interventions
        for comp_idx, component in enumerate(components):
            for layer_win_idx, (layer_start, layer_end) in enumerate(layer_windows):
                for token_win_idx, (token_start, token_end) in enumerate(token_windows):
                    print(f"\nIntervening on {component} from layers {layer_start}-{layer_end} and tokens {token_start}-{token_end}")

                    # Only necessary when using activations but useful to calculate layer window size
                    key_range = [] 
                    for layer_num in range(layer_start, layer_end):
                        key = (component, layer_num)
                        key_range.append(key)

                    # Prepare the intervention configuration
                    intervention_config = IntervenableConfig(
                        representations=[
                            RepresentationConfig(
                                component=component,
                                layer=layer_num,
                            ) for component, layer_num in key_range
                        ],
                        intervention_types=[VanillaIntervention]*len(key_range),
                    )

                    # Initialize the IntervenableModel
                    intervenable_model = IntervenableModel(
                        intervention_config, model
                    )

                    # Prepare unit_locations
                    token_indices = list(range(token_start, token_end))
                    unit_locations = {
                        "sources->base": (
                            [[token_indices]] * len(key_range),  # Positions in source
                            [[token_indices]] * len(key_range),  # Positions in base
                        )
                    }

                    base_input_embeds = {'inputs_embeds': model.model.embed_tokens(inputs['input_ids']).to(device)}
                    noised_input_embeds = copy.deepcopy(noised_inputs).to(device)

                    if base_input_embeds['inputs_embeds'].shape[0] != noised_input_embeds['inputs_embeds'].shape[0]:
                        print("sequence lenghts do not match dim 0")
                        # continue  # Skip this intervention

                    if base_input_embeds['inputs_embeds'].shape[1] != noised_input_embeds['inputs_embeds'].shape[1]:
                        print("sequence lenghts do not match dim 1")
                        # continue  # Skip this intervention

                    restored_output_text = ""
                    for i in range(max_new_tokens):
                        # Run the model with intervention
                        with torch.no_grad():
                            _, restored_outputs = intervenable_model(
                                base=noised_input_embeds,
                                sources=[base_input_embeds] * len(key_range),
                                # source_representations=[activations] * len(key_range),
                                unit_locations = unit_locations,
                            )

                        output_token = torch.argmax(torch.nn.functional.softmax(restored_outputs.logits, dim = -1)[0][-1])
                        output_token_string = tokenizer.decode(output_token, skip_special_tokens=False)

                        print(output_token_string)
                        restored_output_text += output_token_string
                        if output_token_string == "<|eot_id|>":
                            break

                        output_embed = model.model.embed_tokens(torch.tensor([output_token]).to(device))
                        base_input_embeds['inputs_embeds'] = torch.cat(
                            (base_input_embeds['inputs_embeds'], output_embed.unsqueeze(0)), dim=1
                        )
                        noised_input_embeds['inputs_embeds'] = torch.cat(
                            (noised_input_embeds['inputs_embeds'], output_embed.unsqueeze(0)), dim=1
                        )

                    print("Restored Output:")
                    print(restored_output_text)
                    
                    # Decode the output
                    restored_label = label_test_object_in_output(
                        restored_output_text, test_object_line
                    )
                    restored_rule = extract_rule(restored_output_text)

                    # Check if categorization flipped
                    if restored_label is None:
                        continue  # Skip if label not found
                    categorization_flipped = (
                        restored_label == base_label
                    )

                    if categorization_flipped:
                        # Increment flip counts
                        categorization_flip_counts[
                            comp_idx, layer_win_idx, token_win_idx] += 1
                        total_cat_flips += 1

                        # Check if rule also changed
                        if restored_rule == "":
                            continue  # Skip if rule not found
                        rule_changed = not compare_rules(
                            restored_rule, noised_rule
                        )
                        # Increment co-flip counts
                        if rule_changed:
                            co_flip_counts[
                                comp_idx, layer_win_idx, token_win_idx] += 1
                            total_co_flips += 1

                        # Check consistency
                        is_consistent = assess_rule_consistency(
                            restored_label,
                            test_object_line.split(" -> ")[0],
                            restored_rule
                        )
                        # Increment consistency counts
                        if is_consistent:
                            consistency_counts[
                                comp_idx, layer_win_idx,
                                token_win_idx] += 1
                            total_consistencies += 1

                        # Save intervention record
                        intervention_record = {
                            'intervention_component': component,
                            'intervention_layer_window': f"{layer_start}-{layer_end}",
                            'intervention_token_window': f"{token_start}-{token_end}",
                            'rule_changed': rule_changed,
                            'categorization_consistent': is_consistent,
                            'base_label': base_label,
                            'base_rule': base_rule,
                            'noised_label': noised_label,
                            'noised_rule': noised_rule,                                
                            'restored_label': restored_label,
                            'restored_rule': restored_rule,
                        }
                            
                        # Write to CSV
                        with open(f"intervention_records/{rule_name}_interventions.csv", 'a', newline='') as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=intervention_records_fieldnames)
                            writer.writerow(intervention_record)
                            csvfile.flush()
                        
                        # Append to the list
                        intervention_records.append(intervention_record)

            # Save the counts to CSV
            with open(f"intervention_records/{rule_name}_{component}_co_flip_counts.csv", 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(co_flip_counts[comp_idx].astype(int))
            with open(f"intervention_records/{rule_name}_{component}_categorization_flip_counts.csv", 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(categorization_flip_counts[comp_idx].astype(int))
            with open(f"intervention_records/{rule_name}_{component}_consistency_counts.csv", 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(consistency_counts[comp_idx].astype(int))

            # After intervening on each component, calculate partial proportions
            with np.errstate(divide='ignore', invalid='ignore'):
                co_flip_proportions = np.true_divide(
                    co_flip_counts, categorization_flip_counts
                )
                co_flip_proportions[np.isnan(co_flip_proportions)] = 0
                consistency_proportions = np.true_divide(
                    consistency_counts, co_flip_counts
                )
                consistency_proportions[np.isnan(consistency_proportions)] = 0

            # Prepare labels for plotting
            layer_window_labels = [
                f"{start}-{end}" for start, end in layer_windows
            ]
            token_window_labels = [
                f"{start}-{end}" for start, end in token_windows
            ]


            # --- NEW: categorization-flip heatmap ------------------------------------
            plt.figure(figsize=(12, 8))
            plt.imshow(categorization_flip_counts[comp_idx],
                    aspect='auto', cmap='inferno', interpolation='nearest')
            plt.colorbar(label='Count of Categorization Flips')
            plt.xlabel('Token Windows')
            plt.ylabel('Layer Windows')
            plt.title(f'{rule_name}: Categorization Flips\n'
                    f'{component.capitalize()} Component')
            plt.xticks(ticks=range(len(token_window_labels)),
                    labels=token_window_labels, rotation=90, fontsize=6)
            plt.yticks(ticks=range(len(layer_window_labels)),
                    labels=layer_window_labels, fontsize=6)
            plt.tight_layout()
            filename = f"heatmaps/{rule_name}_{component}_categorization_flip.png"
            plt.savefig(filename)
            plt.close()
            log_file.write(f"Categorization-flip heatmap saved as {filename}\n")
            # -------------------------------------------------------------------------

        
            # Plot the results for each component
            plt.figure(figsize=(12, 8))
            plt.imshow(co_flip_proportions[comp_idx],
                       aspect='auto', cmap='viridis',
                       interpolation='nearest')
            plt.colorbar(
                label='Proportion of Rule Changes Given Categorization Flip'
            )
            plt.xlabel('Token Windows')
            plt.ylabel('Layer Windows')
            plt.title(
                f'{rule_name}: Co-Variation of Rule and Categorization\n'
                f'{component.capitalize()} Component'
            )
            plt.xticks(
                ticks=range(len(token_window_labels)),
                labels=token_window_labels,
                rotation=90, fontsize=6
            )
            plt.yticks(ticks=range(len(layer_window_labels)),
                       labels=layer_window_labels, fontsize=6)
            plt.tight_layout()
            # Save the heatmap
            filename = f"heatmaps/{rule_name}_{component}_co_flip.png"
            plt.savefig(filename)
            plt.close()
            log_file.write(f"Co-flip heatmap saved as {filename}\n")
            
            # Plot consistency proportions heatmap
            plt.figure(figsize=(12, 8))
            plt.imshow(consistency_proportions[comp_idx],
                       aspect='auto', cmap='plasma',
                       interpolation='nearest')
            plt.colorbar(
                label='Proportion of Consistent Co-Flips'
            )
            plt.xlabel('Token Windows')
            plt.ylabel('Layer Windows')
            plt.title(
                f'{rule_name}: Consistency of Co-Flips\n'
                f'{component.capitalize()} Component'
            )
            plt.xticks(
                ticks=range(len(token_window_labels)),
                labels=token_window_labels,
                rotation=90, fontsize=6
            )
            plt.yticks(ticks=range(len(layer_window_labels)),
                       labels=layer_window_labels, fontsize=6)
            plt.tight_layout()
            # Save the heatmap
            filename = f"heatmaps/{rule_name}_{component}_consistency.png"
            plt.savefig(filename)
            plt.close()
            log_file.write(f"Consistency heatmap saved as {filename}\n")

        # After interventions, calculate proportions as before
        with np.errstate(divide='ignore', invalid='ignore'):
            co_flip_proportions = np.true_divide(
                co_flip_counts, categorization_flip_counts
            )
            co_flip_proportions[np.isnan(co_flip_proportions)] = 0
            consistency_proportions = np.true_divide(
                consistency_counts, co_flip_counts
            )
            consistency_proportions[np.isnan(consistency_proportions)] = 0

        # Aggregate Statistics
        if total_cat_flips > 0:
            overall_co_flip_proportion = total_co_flips / total_cat_flips
            overall_consistency_proportion = (
                total_consistencies / total_co_flips
                if total_co_flips > 0 else 0
            )
            log_file.write(
                f"For {rule_name}, there were {int(total_cat_flips)} "
                f"interventions resulting in a change of categorization,\n"
            )
            log_file.write(
                f"and {int(total_co_flips)} of these also resulted in a "
                f"corresponding rule change,\n"
            )
            log_file.write(
                "indicating that the rule elicitation co-varies "
                f"{overall_co_flip_proportion:.2%} of the time with "
                "changes of categorizations.\n"
            )
            log_file.write(
                f"Out of the co-flips, {int(total_consistencies)} had "
                "consistent rules and categorizations,\n"
            )
            log_file.write(
                "resulting in a consistency rate of "
                f"{overall_consistency_proportion:.2%} among co-flips.\n"
            )
        else:
            log_file.write(
                f"No categorization flips occurred for {rule_name}.\n"
            )
            
    else:
        log_file.write(
            f"Could not collect both correct and incorrect runs for {rule_name}.\n"
        )

    # Close the log file for this rule
    log_file.close()
