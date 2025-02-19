from utilities import compare_rules, assess_rule_consistency

# Install Pyvene if not already installed
try:
    import pyvene
except ModuleNotFoundError:
    !pip install git+https://github.com/stanfordnlp/pyvene.git

# Import necessary modules
import torch
from pyvene import (
    IntervenableModel,
    IntervenableConfig,
    RepresentationConfig,
    VanillaIntervention,
)
from transformers import GPT2Tokenizer, GPT2LMHeadModel, set_seed
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
# Replace 'gpt2' with a capable model that suits your needs (e.g., 'gpt2-xl')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token exists
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.to(device)
model.eval()

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)

# Create directory to save heatmaps
os.makedirs('heatmaps', exist_ok=True)

# Abbreviations for sizes, colors, and shapes
sizes = ['S', 'M', 'L']       # Small, Medium, Large
colors = ['BLU', 'GRN', 'YEL', 'RED']  # Blue, Green, Yellow, Red
shapes = ['CIR', 'TRI', 'REC', 'SQR']  # Circle, Triangle, Rectangle, Square

# Define the rules and their labeling functions
rules = [
    ("An object is labeled 'True' if it is BLU or a REC.", "Rule1"),
    ("An object is labeled 'True' if it is not a CIR.", "Rule2"),
    ("An object is labeled 'True' if it is RED and not an SQR.", "Rule3"),
]

def label_object(description, rule_desc):
    tokens = description.replace('-', '').split()  # Remove '-' and split
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
                # The label is expected to be provided in the same line or next line
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
        return None  # Label not found
    except Exception:
        return None

def generate_prompt_and_get_token_positions(rule_desc):
    # Define the label pattern
    label_pattern = ['False', 'True', 'True', 'False', 'True', 'False']
    # Generate examples according to the label pattern
    examples = generate_examples_for_rule(rule_desc, label_pattern)
    
    # Build the prompt
    prompt = (
        "Learn the secret rule to label the objects correctly. "
        "If an object follows the rule, it should be labeled 'True'. "
        "Otherwise, it should be labeled 'False'. "
        "Always follow the arrow '->' with an object's label.\n\n"
        "Here are some objects and their labels:\n"
    )
    for example in examples[:5]:  # Use first 5 for the initial examples
        prompt += f"{example}\n"
    prompt += "\nLabel the following object:\n"
    test_object_line = examples[5].split(' -> ')[0]
    prompt += f"{test_object_line}\n"
    prompt += (
        "\nProvide the label for the object above, then explain the rule you've inferred starting with "
        "'Based on your examples, I've learned that'\n"
    )
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    tokenized_prompt = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    return prompt, inputs, tokenized_prompt, test_object_line

def get_output_token_positions(output_sequences, tokenized_prompt, max_new_tokens):
    generated_ids = output_sequences[0][inputs['input_ids'].shape[1]:]
    tokenized_output = tokenizer.convert_ids_to_tokens(generated_ids)

    # We will only consider the first few output tokens (e.g., first 5 tokens)
    output_tokens_to_include = 5  # Adjust as needed
    tokenized_output = tokenized_output[:output_tokens_to_include]
    tokens = tokenized_prompt + tokenized_output

    position_names = [f'Token_{i}' for i in range(len(tokens))]
    token_positions = {name: i for i, name in enumerate(position_names)}

    return tokens, position_names, token_positions

# Total number of layers in the model
num_layers = model.config.n_layer

# Components to intervene on
components = ['residual', 'mlp_activation', 'attention_output']

# Run the experiment for each rule
for rule_idx, (rule_description, rule_name) in enumerate(rules):
    print(f"\n--- Processing {rule_name}: {rule_description} ---\n")
    
    # Initialize results storage for this rule
    prompt, inputs, tokenized_prompt, test_object_line = generate_prompt_and_get_token_positions(rule_description)
    max_new_tokens = 5  # Maximum number of tokens to generate
    print("Prompt:")
    print(prompt)
    
    # Number of correct and incorrect runs to collect
    num_correct_runs = 3
    num_incorrect_runs = 3
    
    # Collect correct and incorrect runs
    correct_runs = []
    incorrect_runs = []
    
    # List to keep track of seeds used
    used_seeds = set()
    
    # Function to capture activations with correct variable scoping
    def get_activation_capturer(activations_dict, key):
        def capturer(module, input, output):
            if key not in activations_dict:
                activations_dict[key] = []
            activations_dict[key].append(output.detach())
        return capturer
    
    # Function to register hooks
    def register_hooks(activations):
        handles = []
        for layer_num in range(num_layers):
            layer = model.transformer.h[layer_num]
            for component in components:
                key = (component, layer_num)
                if component == 'residual':
                    handle = layer.register_forward_hook(get_activation_capturer(activations, key))
                elif component == 'mlp_activation':
                    handle = layer.mlp.register_forward_hook(get_activation_capturer(activations, key))
                elif component == 'attention_output':
                    handle = layer.attn.register_forward_hook(get_activation_capturer(activations, key))
                handles.append(handle)
        return handles
    
    # Collect correct runs
    attempts = 0
    while len(correct_runs) < num_correct_runs and attempts < 100:
        seed = random.randint(0, 10000)
        if seed in used_seeds:
            continue
        used_seeds.add(seed)
        set_seed(seed)
        activations = {}
        handles = register_hooks(activations)
        # Generate output
        output_sequences = model.generate(
            input_ids=inputs['input_ids'],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
        # Remove hooks
        for handle in handles:
            handle.remove()
        output_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        assigned_label = label_test_object_in_output(output_text, test_object_line)
        true_label = label_object(test_object_line, rule_description)
        if assigned_label == true_label and assigned_label is not None:
            # Get combined tokens and positions
            tokens, position_names, token_positions = get_output_token_positions(output_sequences, tokenized_prompt, max_new_tokens)
            correct_runs.append({
                'text': output_text,
                'activations': activations,
                'assigned_label': assigned_label,
                'rule': extract_rule(output_text),
                'tokens': tokens,
                'position_names': position_names,
                'token_positions': token_positions,
                'output_sequences': output_sequences,
            })
        attempts += 1
    
    # Collect incorrect runs
    attempts = 0
    while len(incorrect_runs) < num_incorrect_runs and attempts < 100:
        seed = random.randint(0, 10000)
        if seed in used_seeds:
            continue
        used_seeds.add(seed)
        set_seed(seed)
        activations = {}
        handles = register_hooks(activations)
        # Generate output
        output_sequences = model.generate(
            input_ids=inputs['input_ids'],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
        # Remove hooks
        for handle in handles:
            handle.remove()
        output_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        assigned_label = label_test_object_in_output(output_text, test_object_line)
        true_label = label_object(test_object_line, rule_description)
        if assigned_label is not None and assigned_label != true_label:
            # Get combined tokens and positions
            tokens, position_names, token_positions = get_output_token_positions(output_sequences, tokenized_prompt, max_new_tokens)
            incorrect_runs.append({
                'text': output_text,
                'activations': activations,
                'assigned_label': assigned_label,
                'rule': extract_rule(output_text),
                'tokens': tokens,
                'position_names': position_names,
                'token_positions': token_positions,
                'output_sequences': output_sequences,
            })
        attempts += 1
    
    # Ensure we have enough runs
    if len(correct_runs) < num_correct_runs or len(incorrect_runs) < num_incorrect_runs:
        print(f"Not enough correct and incorrect runs were collected for {rule_name}.")
        continue
    else:
        # Use the token positions and names from the correct runs (assuming they are consistent)
        position_names = correct_runs[0]['position_names']
        token_positions = correct_runs[0]['token_positions']
        num_positions = len(position_names)
        
        # Initialize results storage
        co_flip_counts = np.zeros((len(components), num_layers, num_positions))
        categorization_flip_counts = np.zeros((len(components), num_layers, num_positions))
        consistency_counts = np.zeros((len(components), num_layers, num_positions))
        
        # Initialize counter for interventions skipped due to length mismatch
        interventions_skipped_due_to_length_mismatch = 0
        
        # Create combinations of correct and incorrect runs
        total_cat_flips = 0
        total_co_flips = 0
        total_consistencies = 0
        for correct_run in correct_runs:
            for incorrect_run in incorrect_runs:
                # Ensure tokens and positions align
                if correct_run['position_names'] != incorrect_run['position_names']:
                    continue  # Skip if token positions don't align
                
                # Perform interventions over components, layers, and positions
                for comp_idx, component in enumerate(components):
                    for layer_num in range(num_layers):
                        for pos_idx, position_name in enumerate(position_names):
                            token_pos = token_positions[position_name]
                            # Collect activations for the component, layer, and token position
                            key = (component, layer_num)
                            # Check if activations are available for the required positions
                            if len(correct_run['activations'][key]) <= token_pos or len(incorrect_run['activations'][key]) <= token_pos:
                                continue  # Skip if activations are missing
                            base_activation = correct_run['activations'][key][token_pos]
                            source_activation = incorrect_run['activations'][key][token_pos]
                            
                            # Create intervention configuration
                            intervention_config = IntervenableConfig(
                                representations=[
                                    RepresentationConfig(
                                        layer=layer_num,
                                        component=component,
                                    )
                                ],
                                intervention_types=VanillaIntervention,
                            )

                            # Initialize the IntervenableModel
                            intervenable_model = IntervenableModel(intervention_config, model)

                            # Prepare unit_locations for swapping activations at the token position
                            unit_locations = {
                                "sources->base": (
                                    [[token_pos]],  # Positions in source
                                    [[token_pos]],  # Positions in base
                                )
                            }

                            # Prepare input_ids
                            base_input_ids = torch.cat([inputs['input_ids'][0], correct_run['output_sequences'][0][inputs['input_ids'].shape[1]:]], dim=0).unsqueeze(0)
                            sources_input_ids = torch.cat([inputs['input_ids'][0], incorrect_run['output_sequences'][0][inputs['input_ids'].shape[1]:]], dim=0).unsqueeze(0)

                            # Check if sequence lengths match
                            if base_input_ids.shape[1] != sources_input_ids.shape[1]:
                                interventions_skipped_due_to_length_mismatch += 1
                                continue  # Skip this intervention

                            # Run the model
                            with torch.no_grad():
                                _, intervened_outputs = intervenable_model(
                                    base={'input_ids': base_input_ids.to(device)},
                                    sources={'input_ids': sources_input_ids.to(device)},
                                    activations={
                                        'base': {key: correct_run['activations'][key]},
                                        'sources': {key: incorrect_run['activations'][key]},
                                    },
                                    unit_locations=unit_locations,
                                    max_length=base_input_ids.shape[1],
                                    do_sample=False,  # Keep deterministic to isolate the effect of intervention
                                    temperature=0.7,
                                    pad_token_id=tokenizer.pad_token_id,
                                )

                            # Decode the intervened output
                            intervened_text = tokenizer.decode(intervened_outputs.sequences[0], skip_special_tokens=True)

                            # Extract assigned label and rule from intervened output
                            assigned_label_intervened = label_test_object_in_output(intervened_text, test_object_line)
                            rule_intervened = extract_rule(intervened_text)

                            # Compare to correct run
                            assigned_label_correct = correct_run['assigned_label']
                            rule_correct = correct_run['rule']

                            # Check if categorization flipped
                            if assigned_label_intervened is None:
                                continue  # Skip if label not found
                            categorization_flipped = (assigned_label_intervened != assigned_label_correct)

                            if categorization_flipped:
                                # Increment categorization flip count
                                categorization_flip_counts[comp_idx, layer_num, pos_idx] += 1
                                total_cat_flips += 1

                                # Check if rule also changed compared to correct run
                                if rule_intervened == "":
                                    continue  # Skip if rule not found
                                rule_changed = not compare_rules(rule_intervened, rule_correct)

                                if rule_changed:
                                    # Increment co-flip count
                                    co_flip_counts[comp_idx, layer_num, pos_idx] += 1
                                    total_co_flips += 1
                                    
                                    # Assess consistency between the new assigned label and the new rule
                                    is_consistent = assess_rule_consistency(
                                        assigned_label_intervened,
                                        test_object_line,
                                        rule_intervened
                                    )
                                    if is_consistent:
                                        consistency_counts[comp_idx, layer_num, pos_idx] += 1
                                        total_consistencies += 1

        # After all interventions, report how many were skipped due to length mismatch
        print(f"Interventions skipped due to sequence length mismatch: {interventions_skipped_due_to_length_mismatch}")

        # Calculate the proportions
        with np.errstate(divide='ignore', invalid='ignore'):
            co_flip_proportions = np.true_divide(co_flip_counts, categorization_flip_counts)
            co_flip_proportions[np.isnan(co_flip_proportions)] = 0  # Set NaNs to zero where there were no flips
            consistency_proportions = np.true_divide(consistency_counts, co_flip_counts)
            consistency_proportions[np.isnan(consistency_proportions)] = 0  # Set NaNs to zero where no co-flips occurred

        # Aggregate Statistics
        if total_cat_flips > 0:
            overall_co_flip_proportion = total_co_flips / total_cat_flips
            overall_consistency_proportion = total_consistencies / total_co_flips if total_co_flips > 0 else 0
            print(f"For {rule_name}, there were {int(total_cat_flips)} interventions resulting in a change of categorization,")
            print(f"and {int(total_co_flips)} of these also resulted in a corresponding rule change,")
            print(f"indicating that the rule elicitation co-varies {overall_co_flip_proportion:.2%} of the time with changes of categorizations.")
            print(f"Out of the co-flips, {int(total_consistencies)} had consistent rules and categorizations,")
            print(f"resulting in a consistency rate of {overall_consistency_proportion:.2%} among co-flips.")
        else:
            print(f"No categorization flips occurred for {rule_name}.")

        # Plot the results for each component
        for comp_idx, component in enumerate(components):
            plt.figure(figsize=(15, 6))
            plt.imshow(co_flip_proportions[comp_idx], aspect='auto', cmap='viridis')
            plt.colorbar(label='Proportion of Rule Changes Given Categorization Flip')
            plt.xlabel('Token Position')
            plt.ylabel('Layer')
            plt.title(f'{rule_name}: Co-Variation of Rule and Categorization\n{component.capitalize()} Component')
            plt.xticks(ticks=range(num_positions), labels=position_names, rotation=90, fontsize=6)
            plt.yticks(ticks=range(num_layers))
            plt.tight_layout()
            # Save the heatmap with an informative name
            filename = f"heatmaps/{rule_name}_{component}_co_flip.png"
            plt.savefig(filename)
            plt.close()
            print(f"Co-flip heatmap saved as {filename}")
            
            # Plot the consistency proportions heatmap
            plt.figure(figsize=(15, 6))
            plt.imshow(consistency_proportions[comp_idx], aspect='auto', cmap='plasma')
            plt.colorbar(label='Proportion of Consistent Co-Flips')
            plt.xlabel('Token Position')
            plt.ylabel('Layer')
            plt.title(f'{rule_name}: Consistency of Co-Flips\n{component.capitalize()} Component')
            plt.xticks(ticks=range(num_positions), labels=position_names, rotation=90, fontsize=6)
            plt.yticks(ticks=range(num_layers))
            plt.tight_layout()
            # Save the heatmap with an informative name
            filename = f"heatmaps/{rule_name}_{component}_consistency.png"
            plt.savefig(filename)
            plt.close()
            print(f"Consistency heatmap saved as {filename}")
