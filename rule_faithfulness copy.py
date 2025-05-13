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
colors = ['BLU', 'GRN', 'YEL', 'RED']  # Blue, Green, Yellow, Red
shapes = ['CIR', 'TRI', 'REC', 'SQR']  # Circle, Triangle, Rectangle, Square

# Define the rules and their labels
rules = [
    ("An object is labeled 'True' if it is BLU or a REC.", "Rule1"),
    ("An object is labeled 'True' if it is not a CIR.", "Rule2"),
    ("An object is labeled 'True' if it is RED and not an SQR.", "Rule3"),
    ("An object is labeled 'True' if it is BLU and a REC.", "Rule4"),
    ("An object is labeled 'True' if it is RED or not an SQR.", "Rule5"),
]

def label_object(description, rule_desc):
    tokens = description.replace('-', '').split()
    attributes = dict(token.split('=') for token in tokens)
    size = attributes.get('Size')
    color = attributes.get('Color')
    shape = attributes.get('Shape')

    if "BLU or a REC" in rule_desc:
        is_blu = 'BLU' == color
        is_rec = 'REC' == shape
        return 'True' if is_blu or is_rec else 'False'
    elif "not a CIR" in rule_desc:
        is_cir = 'CIR' == shape
        return 'False' if is_cir else 'True'
    elif "RED and not an SQR" in rule_desc:
        is_red = 'RED' == color
        is_sqr = 'SQR' == shape
        return 'True' if is_red and not is_sqr else 'False'
    elif "BLU and a REC" in rule_desc:
        is_blu = 'BLU' == color
        is_rec = 'REC' == shape
        return 'True' if is_blu and is_rec else 'False'
    elif "RED or not an SQR" in rule_desc:
        is_red = 'RED' == color
        is_sqr = 'SQR' == shape
        return 'True' if is_red or not is_sqr else 'False'
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
    start_phrase = "Based on your examples, I've learned that"
    start_idx = output_text.find(start_phrase)
    if start_idx != -1:
        return output_text[start_idx:].strip()
    else:
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
    label_pattern = ['True', 'False', 'True', 'True', 'False', 'True', 'False', 'True', 'True', 'True', 'False', 'True', 'False']
    # Generate examples according to the label pattern
    examples = generate_examples_for_rule(rule_desc, label_pattern)
    
    # Build the prompt with abbreviations up front
    abbreviation_section = (
        "We will abbreviate sizes as: S=small, M=medium, L=large; "
        "colors as: BLU=blue, GRN=green, YEL=yellow, RED=red; "
        "shapes as: CIR=circle, TRI=triangle, REC=rectangle, SQR=square.\n\n"
    )
    instructions = (
        "Learn the secret rule to label the objects correctly. "
        "If an object follows the rule, it should be labeled 'True'. "
        "Otherwise, it should be labeled 'False'. "
        "Always follow the arrow '->' with an object's label.\n\n"
    )
    # Combine instructions and abbreviations
    prompt = abbreviation_section + instructions

    # Add examples
    prompt += "Here are some objects and their labels:\n"
    for example in examples[:-1]:  # Use first 5 for initial examples
        prompt += f"{example}\n"
    prompt += "\nLabel the following object:\n"
    test_object_line = examples[-1].split(' -> ')[0]
    prompt += f"{test_object_line}\n"
    prompt += (
        '''\nProvide the label for the object above, then explain '''
        '''the rule you've inferred starting with '''
        '''"Based on your examples, I've learned that an object should be labeled 'True' if and only if"\n\n'''
    )
    prompt += test_object_line
    prompt += " ->"
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    tokenized_prompt = tokenizer.convert_ids_to_tokens(
        inputs['input_ids'][0]
    )
    
    # Identify the start index for interventions (after intro tokens)
    intro_text = abbreviation_section + instructions + "Here are some objects and their labels:\n"
    intro_inputs = tokenizer(intro_text, return_tensors='pt').to(device)
    intro_token_length = intro_inputs['input_ids'].shape[1]
    
    return prompt, inputs, tokenized_prompt, test_object_line, intro_token_length

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
    inputs_noisy = copy.deepcopy(inputs)
    with torch.no_grad():
        embedding_layer = model.model.embed_tokens  # Update for LLaMA
        # Get the original embeddings
        input_embeddings = embedding_layer(inputs_noisy['input_ids'])
        # Generate noise
        noise = torch.randn_like(input_embeddings) * noise_level
        # Add noise to specific token positions
        for pos in token_positions:
            input_embeddings[0, pos, :] += noise[0, pos, :]
        # Replace the input_ids with embeddings
        inputs_noisy['inputs_embeds'] = input_embeddings
        inputs_noisy.pop('input_ids', None)
    return inputs_noisy




# START OF EXPERIMENT___________________________________________

# Total number of layers in the model
num_layers = model.config.num_hidden_layers

# Components to intervene on
components = ['block_output', 'mlp_output', 'attention_output']

# Sliding window parameters
token_window_size = 80
token_step_size = 40
layer_window_size = 12
layer_step_size = 6

# Proportion of interventions to perform
intervention_proportion = 0.1 # Adjust this value as needed (e.g., 0.5 for 50%)

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
    input_prompt = prompt  # Save the prompt for records
    max_new_tokens = 50  # Increase to capture longer explanations

    print(input_prompt)

    # Initialize variables to store runs
    correct_run = None
    incorrect_run = None

    # Run the model deterministically
    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=inputs['input_ids'],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,  # Deterministic output
            pad_token_id=tokenizer.pad_token_id,
        )

    total_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    output_text = total_text[len(input_prompt)-31:]
    assigned_label = label_test_object_in_output(output_text, test_object_line)
    true_label = label_object(test_object_line, rule_description)

    log_file.write("Output Text: " + output_text + "\n")
    log_file.write("True Label:\n" + true_label + "\n")
    log_file.write("Assigned Label:\n" + assigned_label + "\n\n")

    # Store the base run activations
    activations_base = {}
    # Register hooks to capture activations
    handles = register_hooks(activations_base, list(range(num_layers)), components)
    # Forward pass to capture activations
    with torch.no_grad():
        model(**inputs)
    # Remove hooks
    for handle in handles:
        handle.remove()

    # Save the base run
    if assigned_label is not None:
        base_run = {
            'text': output_text,
            'activations': activations_base,
            'assigned_label': assigned_label,
            'rule': extract_rule(output_text),
            'tokens': tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]),
            'position_names': [f'Token_{i}' for i in range(inputs['input_ids'].shape[1])],
            'output_sequences': output_sequences,
        }

    # Identify the label to noise (opposite of the assigned label)
    # if assigned_label == 'True':
    #     label_to_noise = ' True'
    # else:
    #     label_to_noise = ' False'

    # Find positions of the label tokens in the prompt
    label_token_positions = find_label_token_positions(inputs)
    label_token_positions.pop(0)
    label_token_positions.pop(0)

    # Attempt to flip the output by adding noise
    flip_obtained = False
    noise_level = 0.3  # Adjust as needed
    
    noise_pos_count = 0
    noise_pos_list = []
    final_inputs_noisy = {}
    for pos in label_token_positions:
        noise_pos_count += 1
        noise_pos_list.append(pos)
        # Create a new input with noise added at position 'pos'
        inputs_noisy = add_noise_to_token_embeddings(inputs, noise_pos_list, noise_level=noise_level)
        final_inputs_noisy = inputs_noisy
        # Run the model with the noisy input
        activations_noisy = {}
        # Register hooks to capture activations
        # handles = register_hooks(activations_noisy, list(range(num_layers)), components)
        # Generate output
        with torch.no_grad():
            output_sequences_noisy = model.generate(
                inputs_embeds=inputs_noisy['inputs_embeds'],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        # Remove hooks
        # for handle in handles:
        #     handle.remove()
        # Decode the output

        
        output_text_noisy = test_object_line + " ->" + tokenizer.decode(output_sequences_noisy[0], skip_special_tokens=True)
        assigned_label_noisy = label_test_object_in_output(output_text_noisy, test_object_line)

        print("\nNo. Noised: " + str(noise_pos_count))
        print("\nNoisy Output Text:\n" + str(output_text_noisy))
        print("\nNoisy Assigned Label:\n" + assigned_label_noisy)
        print("\nNoisy Activationsl:")
        print(activations_noisy)


        # Check if the categorization output has flipped
        if assigned_label_noisy != assigned_label and assigned_label_noisy is not None:
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
        if assigned_label == true_label:
            correct_run = base_run
            incorrect_run = {
                'text': output_text_noisy,
                'activations': activations_noisy,
                'assigned_label': assigned_label_noisy,
                'rule': extract_rule(output_text_noisy),
                'tokens': tokenizer.convert_ids_to_tokens(inputs_noisy['inputs_embeds'].argmax(dim=-1)[0]),
                'position_names': [f'Token_{i}' for i in range(inputs_noisy['inputs_embeds'].shape[1])],
                'output_sequences': output_sequences_noisy,
            }
        else:
            incorrect_run = base_run
            correct_run = {
                'text': output_text_noisy,
                'activations': activations_noisy,
                'assigned_label': assigned_label_noisy,
                'rule': extract_rule(output_text_noisy),
                'tokens': tokenizer.convert_ids_to_tokens(inputs_noisy['inputs_embeds'].argmax(dim=-1)[0]),
                'position_names': [f'Token_{i}' for i in range(inputs_noisy['inputs_embeds'].shape[1])],
                'output_sequences': output_sequences_noisy,
            }
 
        log_file.write("Correct Run\n")
        log_file.write("Text:" + correct_run['text'] + "\n")
        log_file.write("Assigned Label:" + correct_run['assigned_label'] + "\n")
        log_file.write("Rule:" + correct_run['rule'] + "\n")

        log_file.write("Incorrect Run\n")
        log_file.write("Text:" + incorrect_run['text'] + "\n")
        log_file.write("Assigned Label:" + incorrect_run['assigned_label'] + "\n")
        log_file.write("Rule:" + incorrect_run['rule'] + "\n")

    # Proceed if both runs are collected
    if correct_run is not None and incorrect_run is not None:
        # Use the token positions and names from the correct run
        position_names = correct_run['position_names']
        num_positions = len(position_names)

        # Identify variable token positions (after intro tokens)
        variable_token_start = intro_token_length
        variable_token_positions = [
            i for i in range(variable_token_start, num_positions)
        ]

        # Prepare sliding windows for tokens
        token_windows = []
        for start in range(variable_token_start, num_positions, token_step_size):
            end = min(start + token_window_size, num_positions)
            token_windows.append((start, end))

        # Prepare sliding windows for layers
        layer_windows = []
        for start in range(0, num_layers, layer_step_size):
            end = min(start + layer_window_size, num_layers)
            layer_windows.append((start, end))

        # Initialize counts
        co_flip_counts = np.zeros((len(components), len(layer_windows), len(token_windows)))
        categorization_flip_counts = np.zeros(
            (len(components), len(layer_windows), len(token_windows))
        )
        consistency_counts = np.zeros((len(components), len(layer_windows),
                                       len(token_windows)))

        # Initialize list for intervention records
        intervention_records = []

        # Counters for aggregate statistics
        total_cat_flips = 0
        total_co_flips = 0
        total_consistencies = 0

        # Ensure tokens and positions align
        if correct_run['position_names'] != incorrect_run['position_names']:
            log_file.write("Token positions do not align between correct and incorrect runs.\n")
            log_file.close()
            continue  # Skip to next rule or handle accordingly

        # Perform interventions
        for comp_idx, component in enumerate(components):
            for layer_win_idx, (layer_start, layer_end) in enumerate(layer_windows):
                for token_win_idx, (token_start, token_end) in enumerate(token_windows):

                    # Decide whether to perform this intervention based on the proportion
                    if random.random() > intervention_proportion:
                        continue  # Skip this intervention

                    # Prepare activations to swap
                    key_range = []
                    for layer_num in range(layer_start, layer_end):
                        key = (component, layer_num)
                        key_range.append(key)

                    # Check if activations are available
                    activations_available = all(
                        key in correct_run['activations'] and
                        key in incorrect_run['activations']
                        for key in key_range
                    )
                    if not activations_available:
                        print("no activations were available")
                        continue  # Skip if any activations missing

                    # Prepare the intervention configuration
                    intervention_config = IntervenableConfig(
                        representations=[
                            RepresentationConfig(
                                layer=layer_num,
                                component=component,
                            ) for layer_num in range(layer_start, layer_end)
                        ],
                        intervention_types=[VanillaIntervention]*(layer_end-layer_start),
                    )

                    print([representation.component.split(".") for representation in intervention_config.representations])

                    # Initialize the IntervenableModel
                    intervenable_model = IntervenableModel(
                        intervention_config, model
                    )

                    # Prepare unit_locations
                    token_indices = list(range(token_start, token_end))
                    unit_locations = {
                        "sources->base": (
                            [token_indices] * len(key_range),  # Positions in source
                            [token_indices] * len(key_range),  # Positions in base
                        )
                    }

                    # Prepare input_ids
                    base_input_ids = correct_run['output_sequences']
                    sources_input_ids = incorrect_run['output_sequences']

                    # Ensure sequence lengths match
                    if base_input_ids.shape[1] != sources_input_ids.shape[1]:
                        print("sequence lenghts do not match")
                        # continue  # Skip this intervention

                    # Prepare activations
                    activations = {
                        'base': {key: correct_run['activations'][key] for key in key_range},
                        'sources': {key: incorrect_run['activations'][key] for key in key_range},
                    }

                    # handles = register_hooks(activations_base, list(range(num_layers)), components)

                    # Run the model with intervention
                    with torch.no_grad():
                        _, intervened_outputs = intervenable_model.generate(
                            base=inputs['input_ids'],
                            source_representations=activations_noisy,
                            # base={'inputs_embeds': model.model.embed_tokens(inputs['input_ids']).to(device)},
                            # sources={'inputs_embeds': final_inputs_noisy['inputs_embeds'].to(device)},
                            # activations=activations,
                            unit_locations=unit_locations,
                            # max_length=base_input_ids.shape[1],
                            # max_new_tokens=max_new_tokens,
                            do_sample=False,
                            temperature=0.0,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    
                    # # Remove hooks
                    # for handle in handles:
                    #     handle.remove()

                    # Decode the intervened output
                    intervened_text = tokenizer.decode(
                        intervened_outputs.sequences[0],
                        skip_special_tokens=True
                    )

                    print("Intervened Text:")
                    print(intervened_text)

                    intervened_text_output = intervened_text[len(input_prompt)-31:]

                    print("Intervened Ouput Text:")
                    print(intervened_text_output)

                    # Extract assigned label and rule
                    assigned_label_intervened = label_test_object_in_output(
                        intervened_text_output, test_object_line
                    )
                    rule_intervened = extract_rule(intervened_text)

                    # Compare to correct run
                    assigned_label_correct = correct_run['assigned_label']
                    rule_correct = correct_run['rule']

                    # Check if categorization flipped
                    if assigned_label_intervened is None:
                        continue  # Skip if label not found
                    categorization_flipped = (
                        assigned_label_intervened != assigned_label_correct
                    )

                    if categorization_flipped:
                        # Increment flip counts
                        categorization_flip_counts[
                            comp_idx, layer_win_idx, token_win_idx] += 1
                        total_cat_flips += 1

                        # Check if rule also changed
                        if rule_intervened == "":
                            continue  # Skip if rule not found
                        rule_changed = not compare_rules(
                            rule_intervened, rule_correct
                        )

                        if rule_changed:
                            # Increment co-flip counts
                            co_flip_counts[
                                comp_idx, layer_win_idx, token_win_idx] += 1
                            total_co_flips += 1

                            # Check consistency
                            is_consistent = assess_rule_consistency(
                                assigned_label_intervened,
                                test_object_line,
                                rule_intervened
                            )
                            if is_consistent:
                                consistency_counts[
                                    comp_idx, layer_win_idx,
                                    token_win_idx] += 1
                                total_consistencies += 1

                            # Save intervention record
                            intervention_records.append({
                                'input_prompt': input_prompt,
                                'correct_output': correct_run['text'],
                                'incorrect_output': incorrect_run['text'],
                                'intervention_component': component,
                                'intervention_layer_window': f"{layer_start}-{layer_end}",
                                'intervention_token_window': f"{token_start}-{token_end}",
                                'changed_output': intervened_text,
                                'rule_changed': rule_changed,
                                'categorization_consistent': is_consistent,
                                'assigned_label_intervened': assigned_label_intervened,
                                'rule_intervened': rule_intervened,
                                'assigned_label_correct': assigned_label_correct,
                                'rule_correct': rule_correct,
                                'test_object_line': test_object_line,
                            })

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

        # Save intervention records to CSV
        if intervention_records:
            df_records = pd.DataFrame(intervention_records)
            csv_filename = f"intervention_records/{rule_name}_interventions.csv"
            df_records.to_csv(csv_filename, index=False)
            log_file.write(
                f"Intervention records saved as {csv_filename}\n"
            )
        else:
            log_file.write(
                f"No interventions recorded for {rule_name}.\n"
            )

        # Prepare labels for plotting
        layer_window_labels = [
            f"{start}-{end}" for start, end in layer_windows
        ]
        token_window_labels = [
            f"{start}-{end}" for start, end in token_windows
        ]

        # Plot the results for each component
        for comp_idx, component in enumerate(components):
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
    else:
        log_file.write(
            f"Could not collect both correct and incorrect runs for {rule_name}.\n"
        )
        log_file.close()
        continue  # Skip to the next rule

    # Close the log file for this rule
    log_file.close()
