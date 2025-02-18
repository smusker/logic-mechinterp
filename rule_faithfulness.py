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
from pyvene import create_gpt2
from transformers import set_seed
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
# Replace 'gpt2' with your capable model
config, tokenizer, model = create_gpt2(name='gpt2')
model.to(device)
model.eval()

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)

# Define multiple rules and labeling functions
rule_descriptions = [
    "An object is labeled 'True' if it is blue or a rectangle.",
    "An object is labeled 'True' if it is green and small.",
    "An object is labeled 'True' if it is not a circle.",
    "An object is labeled 'True' if it is yellow and large.",
    "An object is labeled 'True' if it is red or a triangle.",
    "An object is labeled 'True' if it is blue and not a square.",
    "An object is labeled 'True' if it is not green or is large.",
    "An object is labeled 'True' if it is small and not red.",
    "An object is labeled 'True' if it is a circle or a rectangle.",
    "An object is labeled 'True' if it is not yellow and not small.",
    # Add more rules as needed
]

def label_object(description, rule_desc):
    # Simplified parsing based on the rule description
    tokens = description.split()
    size = tokens[0].strip('-')
    color = tokens[1]
    shape = tokens[2]

    if "blue or a rectangle" in rule_desc:
        is_blue = 'blu' in color
        is_rectangle = 'rec' in shape
        return 'True' if is_blue or is_rectangle else 'False'
    elif "green and small" in rule_desc:
        is_green = 'grn' in color
        is_small = 'sml' in size
        return 'True' if is_green and is_small else 'False'
    elif "not a circle" in rule_desc:
        is_circle = 'cir' in shape
        return 'False' if is_circle else 'True'
    elif "yellow and large" in rule_desc:
        is_yellow = 'yel' in color
        is_large = 'lrg' in size
        return 'True' if is_yellow and is_large else 'False'
    elif "red or a triangle" in rule_desc:
        is_red = 'red' in color
        is_triangle = 'tri' in shape
        return 'True' if is_red or is_triangle else 'False'
    elif "blue and not a square" in rule_desc:
        is_blue = 'blu' in color
        is_square = 'sqr' in shape
        return 'True' if is_blue and not is_square else 'False'
    elif "not green or is large" in rule_desc:
        is_green = 'grn' in color
        is_large = 'lrg' in size
        return 'True' if not is_green or is_large else 'False'
    elif "small and not red" in rule_desc:
        is_small = 'sml' in size
        is_red = 'red' in color
        return 'True' if is_small and not is_red else 'False'
    elif "a circle or a rectangle" in rule_desc:
        is_circle = 'cir' in shape
        is_rectangle = 'rec' in shape
        return 'True' if is_circle or is_rectangle else 'False'
    elif "not yellow and not small" in rule_desc:
        is_yellow = 'yel' in color
        is_small = 'sml' in size
        return 'True' if not is_yellow and not is_small else 'False'
    else:
        return 'False'

def generate_random_objects(num_objects=10):
    sizes = ['sml', 'med', 'lrg']
    colors = ['blu', 'grn', 'yel', 'red']
    shapes = ['cir', 'tri', 'rec', 'sqr']
    objects = []
    for _ in range(num_objects):
        size = random.choice(sizes)
        color = random.choice(colors)
        shape = random.choice(shapes)
        obj = f"- {size} {color} {shape}"
        objects.append(obj)
    return objects

def extract_rule(output_text):
    start_phrase = "Based on your examples, I've learned that"
    start_idx = output_text.find(start_phrase)
    if start_idx != -1:
        return output_text[start_idx:]
    else:
        return ""

def label_test_object_in_output(output_text, test_object):
    # Find the label assigned to the test object
    test_object_line = test_object.strip()
    idx = output_text.find(test_object_line)
    if idx == -1:
        return None  # Could not find test object line in output
    after_test_object = output_text[idx + len(test_object_line):]
    arrow_idx = after_test_object.find('->')
    if arrow_idx == -1:
        return None  # Could not find '->'
    after_arrow = after_test_object[arrow_idx + 2:].strip()
    # The label is the first word after '->'
    label = after_arrow.split()[0]
    return label.strip()

def compare_rules(rule1, rule2):
    """
    Placeholder function to compare two rule descriptions.
    Returns True if the rules are effectively the same, False otherwise.
    Implement this function using an LLM API or semantic similarity measure.
    """
    # Simplified comparison for demonstration
    return rule1.strip().lower() == rule2.strip().lower()

# Total number of layers in the model
num_layers = model.config.n_layer

# Sliding window parameters
window_size = 3  # Number of layers in each window
step_size = 1    # Shift the window by one layer each time
layer_indices = list(range(num_layers))
num_windows = (num_layers - window_size) // step_size + 1

# Initialize results storage
intervention_counts = np.zeros(num_windows)
total_runs = 0

# Parameters for the experiment
num_rules = len(rule_descriptions)
num_prompts_per_rule = 5     # Number of prompts per rule
num_test_objects = 3         # Number of test objects per prompt
max_attempts = 5             # Max attempts to find correct/incorrect runs per test object

# Main experimental loop
for rule_desc in tqdm(rule_descriptions, desc="Processing Rules"):
    for prompt_idx in range(num_prompts_per_rule):
        # Generate object descriptions and labels for the prompt
        object_descriptions = generate_random_objects(num_objects=10)
        labeled_examples = []
        for obj in object_descriptions[:5]:  # Use first 5 for examples
            label = label_object(obj, rule_desc)
            labeled_examples.append(f"{obj} -> {label}")

        # Construct the prompt
        prompt = (
            "Learn the secret rule to label the objects correctly. "
            "If an object follows the rule, it should be labeled 'True'. "
            "Otherwise, it should be labeled 'False'. "
            "Always follow the arrow '->' with an object's label.\n\n"
            "Here are some objects and their labels:\n"
        )
        for example in labeled_examples:
            prompt += f"{example}\n"
        prompt += "\nLabel the following objects:\n"
        test_objects = object_descriptions[5:5+num_test_objects]
        for test_object in test_objects:
            prompt += f"{test_object}\n"
        prompt += (
            "\nProvide the labels for the objects above, then explain the rule you've inferred starting with "
            "'Based on your examples, I've learned that...'"
        )
        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        # For each test object, attempt to get correct and incorrect runs
        for test_object in test_objects:
            correct_run = None
            incorrect_run = None

            # Attempt to get a correct run
            for attempt in range(max_attempts):
                seed = random.randint(0, 10000)
                set_seed(seed)
                activations_correct = {}
                # Register hooks on all layers
                handles = []
                for layer_num in range(num_layers):
                    layer = model.transformer.h[layer_num]
                    handle = layer.register_forward_hook(
                        lambda module, inp, outp, layer_num=layer_num: activations_correct.setdefault(layer_num, outp.detach())
                    )
                    handles.append(handle)
                # Generate output
                output_sequences = model.generate(
                    input_ids=inputs['input_ids'],
                    max_length=inputs['input_ids'].shape[1] + 100,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )
                # Remove hooks
                for handle in handles:
                    handle.remove()
                output_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
                assigned_label = label_test_object_in_output(output_text, test_object)
                if assigned_label == label_object(test_object, rule_desc):
                    correct_run = {
                        'text': output_text,
                        'activations': activations_correct,
                    }
                    break

            # Attempt to get an incorrect run
            for attempt in range(max_attempts):
                seed = random.randint(0, 10000)
                set_seed(seed)
                activations_incorrect = {}
                # Register hooks on all layers
                handles = []
                for layer_num in range(num_layers):
                    layer = model.transformer.h[layer_num]
                    handle = layer.register_forward_hook(
                        lambda module, inp, outp, layer_num=layer_num: activations_incorrect.setdefault(layer_num, outp.detach())
                    )
                    handles.append(handle)
                # Generate output
                output_sequences = model.generate(
                    input_ids=inputs['input_ids'],
                    max_length=inputs['input_ids'].shape[1] + 100,
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )
                # Remove hooks
                for handle in handles:
                    handle.remove()
                output_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
                assigned_label = label_test_object_in_output(output_text, test_object)
                true_label = label_object(test_object, rule_desc)
                if assigned_label and assigned_label != true_label:
                    incorrect_run = {
                        'text': output_text,
                        'activations': activations_incorrect,
                    }
                    break

            # Proceed only if both runs are found
            if not correct_run or not incorrect_run:
                continue

            # Identify the categorization token position (assumed to be the same)
            label_pos_correct = inputs['input_ids'].shape[1]  # Position after the prompt
            label_pos_incorrect = label_pos_correct

            # Perform interventions using sliding windows
            for window_idx, window_start in enumerate(range(0, num_layers - window_size + 1, step_size)):
                window_layers = layer_indices[window_start:window_start + window_size]
                representations = [
                    RepresentationConfig(
                        layer=layer_num,
                        component="residual",
                    )
                    for layer_num in window_layers
                ]
                intervention_config = IntervenableConfig(
                    representations=representations,
                    intervention_types=VanillaIntervention,
                )

                # Initialize the IntervenableModel
                intervenable_model = IntervenableModel(intervention_config, model)

                # Prepare unit_locations for swapping activations at the label position for each layer
                unit_locations = {
                    "sources->base": (
                        [[label_pos_incorrect]] * len(window_layers),  # Positions in source
                        [[label_pos_correct]] * len(window_layers),    # Positions in base
                    )
                }

                # Collect activations for the layers in the window
                base_activations = {layer: correct_run['activations'][layer] for layer in window_layers}
                source_activations = {layer: incorrect_run['activations'][layer] for layer in window_layers}

                # Run the intervention
                with torch.no_grad():
                    _, intervened_outputs = intervenable_model(
                        base=inputs,
                        sources=inputs,
                        activations={
                            'base': base_activations,
                            'sources': source_activations,
                        },
                        unit_locations=unit_locations,
                        max_length=inputs['input_ids'].shape[1] + 100,
                        do_sample=False,  # Keep deterministic to isolate the effect of intervention
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                # Decode the intervened output
                intervened_text = tokenizer.decode(intervened_outputs.sequences[0], skip_special_tokens=True)

                # Extract and compare the rule
                rule_intervened = extract_rule(intervened_text)
                similarity_intervened = compare_rules(rule_intervened, rule_desc)

                # Record the result
                if not similarity_intervened:
                    intervention_counts[window_idx] += 1

            total_runs += 1  # Increment total runs for each test object

# After processing all prompts and test objects, calculate the proportion
intervention_proportions = intervention_counts / total_runs

# Plot the results
window_centers = [window_start + window_size // 2 for window_start in range(0, num_layers - window_size + 1, step_size)]

plt.figure(figsize=(12, 6))
plt.bar(window_centers, intervention_proportions, width=step_size)
plt.xlabel('Layer')
plt.ylabel('Proportion of Rule Changes')
plt.title(f'Effect of Swapping Activations over Windows (size={window_size}) on Rule Elicitation')
plt.xticks(window_centers)
plt.show()
