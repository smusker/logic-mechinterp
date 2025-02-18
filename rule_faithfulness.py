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
    embed_to_distrib,
    top_vals
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
        "Begin your answer with 'Based on your examples, I've learned that...' "
        "and give your best concise description of the rule without asking for more evidence. "
        "Always follow the arrow '->' with an object's label.\n\n"
        "Label the following objects:\n"
        "- med yel tri\n"
        "- sml blu cir\n"
        "- lrg grn rec\n"
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

# Identify the positions to intervene (token positions of labels)
# Assuming that the labels start immediately after the prompt tokens
prompt_length = inputs['input_ids'].shape[1]
# For simplicity, we assume the model outputs the labels in the same order

def find_label_positions(output_text):
    # Tokenize the output
    output_tokens = tokenizer.tokenize(output_text)
    # Find positions of '->' and the label tokens
    label_positions = []
    for idx, token in enumerate(output_tokens):
        if token == 'Ġ->':  # The tokenizer adds 'Ġ' before '->'
            # The label token is expected to be right after '->'
            label_pos = idx + 1
            if label_pos < len(output_tokens):
                label_positions.append(label_pos)
    # Adjust positions to account for the prompt length
    label_positions = [pos + prompt_length for pos in label_positions]
    return label_positions

label_positions_correct = find_label_positions(output_correct)
label_positions_incorrect = find_label_positions(output_incorrect)

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
        [label_positions_incorrect],  # Positions in source to take activations from
        [label_positions_correct],    # Positions in base to overwrite
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

if intervened_text != output_correct:
    print("\nThe rule elicitation changed after swapping the categorization activations.")
else:
    print("\nThe rule elicitation did not change after swapping the categorization activations.")
