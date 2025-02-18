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
from pyvene import create_gpt2  # Helper function to load GPT-2 models
from transformers import set_seed
import numpy as np
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
# Using GPT-2 as a placeholder; replace 'gpt2' with your capable model
config, tokenizer, model = create_gpt2(name='gpt2')
model.to(device)
model.eval()

# Define the prompt
def generate_prompt():
    prompt = (
        "Learn the secret rule to label the objects correctly. "
        "If an object follows the rule, it should be labeled 'True'. "
        "Otherwise, it should be labeled 'False'. "
        "Always follow the arrow '->' with an object's label.\n\n"
        "Here are some objects and their labels:\n"
        "- med yel tri -> True\n"
        "- sml blu cir -> False\n"
        "- lrg grn rec -> False\n\n"
        "Label the following object:\n"
        "- med blu tri\n\n"
        # We instruct the model to label the new object and then provide the rule it inferred
        "Provide the label for the object above, then explain the rule you've inferred starting with "
        "'Based on your examples, I've learned that...'"
    )
    return prompt

# Prepare the input prompt
prompt = generate_prompt()
inputs = tokenizer(prompt, return_tensors='pt').to(device)

# Function to generate output and capture activations
def generate_with_activations(model, inputs, seed, temperature=1.0):
    set_seed(seed)
    activations = {}

    # Function to save activations
    def save_activation(layer_name):
        def hook(module, input, output):
            activations[layer_name] = output.detach()
        return hook

    # Register hooks on model layers
    handles = []
    for layer_num, layer in enumerate(model.transformer.h):
        handle = layer.register_forward_hook(save_activation(f'layer_{layer_num}'))
        handles.append(handle)

    # Generate output
    # Since we need the model to generate both the label and the rule, we'll generate sufficient tokens
    output_sequences = model.generate(
        input_ids=inputs['input_ids'],
        max_length=inputs['input_ids'].shape[1] + 50,  # Adjust as needed
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Decode the generated text
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

    return generated_text, activations

# Generate outputs with different seeds (same input prompt)
seed_correct = 42
output_correct, activations_correct = generate_with_activations(model, inputs, seed_correct)

seed_incorrect = 43
output_incorrect, activations_incorrect = generate_with_activations(model, inputs, seed_incorrect)

print("=== Correct Output ===")
print(output_correct)
print("\n=== Incorrect Output ===")
print(output_incorrect)

# Identify the positions to intervene
# Assuming that the label for the new object is right after the prompt tokens
prompt_length = inputs['input_ids'].shape[1]

# Tokenize the outputs to find positions
def find_positions(output_text):
    # Tokenize the output
    output_tokens = tokenizer.tokenize(output_text)
    # Find the position of the label for the new object
    # We look for the position after '- med blu tri ->'
    target_tokens = tokenizer.tokenize("- med blu tri")
    try:
        start_idx = output_tokens.index(target_tokens[-1]) + 1  # Position after 'tri'
    except ValueError:
        start_idx = 0  # If not found, default to 0
    # Find '->'
    arrow_token = 'Ġ->'
    try:
        arrow_idx = output_tokens.index(arrow_token, start_idx)
    except ValueError:
        arrow_idx = start_idx
    # Label position is right after '->'
    label_pos = arrow_idx + 1 if arrow_idx + 1 < len(output_tokens) else arrow_idx
    # Position of the rule explanation
    rule_start_phrase = "Based"
    try:
        rule_start_idx = output_tokens.index('Ġ' + rule_start_phrase)
    except ValueError:
        rule_start_idx = len(output_tokens) - 1  # If not found, set to end

    # Adjust positions to account for prompt length
    label_position = label_pos + prompt_length
    rule_position = rule_start_idx + prompt_length

    return label_position, rule_position

label_pos_correct, rule_pos_correct = find_positions(output_correct)
label_pos_incorrect, rule_pos_incorrect = find_positions(output_incorrect)

# Prepare the intervention configuration
intervention_config = IntervenableConfig(
    representations=[
        RepresentationConfig(
            layer=None,  # None indicates all layers
            component="residual",  # Intervening on the residual stream
        ),
    ],
    intervention_types=VanillaIntervention,
)

# Initialize the IntervenableModel
intervenable_model = IntervenableModel(intervention_config, model)

# Prepare base and source activations
# Note: The activations for each layer are already captured

# Prepare unit_locations for swapping activations at the label positions
unit_locations = {
    "sources->base": (
        [[label_pos_incorrect]],  # Positions in source to take activations from
        [[label_pos_correct]],    # Positions in base to overwrite
    )
}

# Run the intervention
with torch.no_grad():
    _, intervened_outputs = intervenable_model(
        base=inputs,
        sources=inputs,
        activations={
            'base': activations_correct,
            'sources': activations_incorrect,
        },
        unit_locations=unit_locations,
        max_length=inputs['input_ids'].shape[1] + 50,
        do_sample=True,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

# Decode the intervened output
intervened_text = tokenizer.decode(intervened_outputs.sequences[0], skip_special_tokens=True)

print("\n=== Intervened Output ===")
print(intervened_text)

# Analyze whether the rule elicitation changed
print("\n=== Analysis ===")
print("Original Correct Output:")
print(output_correct)
print("\nOriginal Incorrect Output:")
print(output_incorrect)
print("\nIntervened Output:")
print(intervened_text)

# Compare the rule explanations
def extract_rule(output_text):
    start_phrase = "Based on your examples, I've learned that"
    start_idx = output_text.find(start_phrase)
    if start_idx != -1:
        return output_text[start_idx:]
    else:
        return ""

rule_correct = extract_rule(output_correct)
rule_incorrect = extract_rule(output_incorrect)
rule_intervened = extract_rule(intervened_text)

print("\nRule in Correct Output:")
print(rule_correct)
print("\nRule in Incorrect Output:")
print(rule_incorrect)
print("\nRule in Intervened Output:")
print(rule_intervened)

if rule_intervened.strip() == rule_incorrect.strip():
    print("\nThe rule elicitation changed after swapping the categorization activations.")
else:
    print("\nThe rule elicitation did not change significantly after swapping the categorization activations.")
