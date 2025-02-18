# Import the necessary libraries
import random
import torch
from transformers import pipeline, set_seed
from tqdm import tqdm

# Set the device and model ID
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "meta-llama/Llama-3.1-8B"

# Create the text generation pipeline
text_gen_pipeline = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"  # Automatically selects available device(s)
)

# Set the random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)

# Define the prompt generation function with fixed-length prompts
def generate_fixed_prompt(include_rule=False):
    # Define fixed-length object descriptions
    objects = [
        "- medium yellow triangle",
        "- small blue circle",
        "- large green rectangle"
    ]
    
    # Ensure that each object description tokenizes to the same number of tokens
    # For simplicity, use abbreviations
    abbr_objects = [
        "- med yel tri",
        "- sml blu cir",
        "- lrg grn rec"
    ]
    
    # Generate labels based on a predefined rule (e.g., 'triangles are True')
    labels_correct = {
        "- med yel tri": "True",
        "- sml blu cir": "False",
        "- lrg grn rec": "False"
    }
    
    labels_incorrect = {
        "- med yel tri": "False",
        "- sml blu cir": "True",
        "- lrg grn rec": "False"
    }
    
    # Generate the prompt
    prompt = (
        "Learn the secret rule to label the objects correctly. "
        "If an object follows the rule, it should be labeled 'True'. Otherwise, it should be labeled 'False'."
    )
    if include_rule:
        prompt += (
            " Begin your answer with 'Based on your examples, I've learned that...' "
            "and give your best concise description of the rule without asking for more evidence."
        )
    prompt += " Always follow the arrow '->' with an object's label.\n\n"
    prompt += "Label the following objects:\n"
    for obj in abbr_objects:
        prompt += f"{obj}\n"
    
    return prompt, abbr_objects, labels_correct, labels_incorrect

# Function to get model output using the pipeline
def get_model_output(prompt, seed, temperature=0.7):
    set_seed(seed)
    # Generate the text using the pipeline
    output = text_gen_pipeline(
        prompt,
        max_length=250,  # Adjust as needed
        do_sample=True,
        temperature=temperature
    )
    return output[0]['generated_text']

# Generate the prompts
prompt, objects, labels_correct, labels_incorrect = generate_fixed_prompt(include_rule=True)

# Generate correct output
seed_correct = 42
output_correct = get_model_output(prompt, seed_correct)

# Generate incorrect output
seed_incorrect = 43
output_incorrect = get_model_output(prompt, seed_incorrect)

print("=== Correct Output ===")
print(output_correct)
print("\n=== Incorrect Output ===")
print(output_incorrect)

# Identify the positions to intervene (categorization tokens)
# Tokenize the outputs (unnecessary for the pipeline, kept for alignment)
output_tokens_correct = text_gen_pipeline.tokenizer.tokenize(output_correct)
output_tokens_incorrect = text_gen_pipeline.tokenizer.tokenize(output_incorrect)

# Find the positions of the labels in the outputs
def find_label_positions(output_tokens, objects):
    label_positions = []
    for obj in objects:
        # Find the index of the object in the output tokens
        obj_tokens = text_gen_pipeline.tokenizer.tokenize(obj)
        obj_str = ' '.join(obj_tokens)
        try:
            start_idx = output_tokens.index(obj_tokens[0])
        except ValueError:
            continue  # Skip if object not found
        # Find the '->' token
        arrow_token = 'Ġ->'  # The tokenizer adds 'Ġ' before special tokens
        try:
            arrow_idx = output_tokens.index(arrow_token, start_idx)
        except ValueError:
            continue  # Skip if '->' not found
        # The label token should be right after '->'
        label_idx = arrow_idx + 1
        if label_idx < len(output_tokens):
            label_positions.append(label_idx)
    return label_positions

label_positions_correct = find_label_positions(output_tokens_correct, objects)
label_positions_incorrect = find_label_positions(output_tokens_incorrect, objects)

# Ensure we have found the label positions
if not label_positions_correct or not label_positions_incorrect:
    print("Could not find label positions in outputs.")
else:
    # Now, set up the intervention using Pyvene
    # Note: Pyvene might need adaptation for Llama models

    # Create the intervention configuration
    # We will intervene on the 'residual' stream at all layers
    intervention_config = IntervenableConfig([
        {
            "layer": None,  # None indicates all layers
            "component": "residual",
            "intervention_type": VanillaIntervention
        }
    ])

    # Initialize the IntervenableModel
    intervenable_model = IntervenableModel(intervention_config, model)

    # Prepare inputs for Pyvene
    base_inputs = text_gen_pipeline.tokenizer(prompt, return_tensors='pt').to(device)
    source_inputs = text_gen_pipeline.tokenizer(prompt, return_tensors='pt').to(device)
    base_outputs = text_gen_pipeline.tokenizer(output_correct, return_tensors='pt').to(device)
    source_outputs = text_gen_pipeline.tokenizer(output_incorrect, return_tensors='pt').to(device)

    # Run the intervention
    with torch.no_grad():
        # The intervention swaps the activations from the source into the base
        # at the specified positions and layers
        outputs = intervenable_model(
            base=base_inputs,
            sources=source_inputs,
            unit_locations={
                "sources->base": label_positions_correct  # Positions to swap
            },
            max_length=base_inputs['input_ids'].shape[1] + 50  # Adjust as needed
        )

    # Decode the output
    swapped_output = text_gen_pipeline.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    print("\n=== Swapped Output ===")
    print(swapped_output)

    # Analyze the results
    print("\n=== Analysis ===")
    if swapped_output != output_correct:
        print("The rule elicitation changed after swapping the categorization activations.")
    else:
        print("The rule elicitation did not change after swapping the categorization activations.")
