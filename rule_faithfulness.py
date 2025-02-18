# Import necessary libraries and modules
import torch
import random
from transformers import pipeline, set_seed
from tqdm import tqdm

# Set the device and model identifier
model_id = "meta-llama/Llama-3.1-8B"

# Create the text generation pipeline
text_gen_pipeline = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"  # Automatically selects available device(s)
)

# Set random seeds for reproducibility
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
        max_length=250,  # Adjust max_length as needed
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

# Note: The intervention part using Pyvene requires compatibility checks
# Here, we'll focus on setting up for intervention but actual usage
# depends on Pyvene's compatibility with current transformer architectures

# Reminder: Replace the below with Pyvene logic if applicable

print("\n=== Analysis ===")
if output_correct != output_incorrect:
    print("The outputs are different, indicating a potential change in rule elicitation based on seed.")
else:
    print("The rule elicitation did not change after swapping outputs using different seeds.")
