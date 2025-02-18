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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
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
    # Add more rules as needed
]

def label_object(description, rule_desc):
    # Implement parsing based on the rule description
    if "blue or a rectangle" in rule_desc:
        is_blue = 'blu' in description
        is_rectangle = 'rec' in description
        return 'True' if is_blue or is_rectangle else 'False'
    elif "green and small" in rule_desc:
        is_green = 'grn' in description
        is_small = 'sml' in description
        return 'True' if is_green and is_small else 'False'
    elif "not a circle" in rule_desc:
        is_circle = 'cir' in description
        return 'False' if is_circle else 'True'
    # Add more rules as needed
    else:
        return 'False'

# Total number of layers in the model
num_layers = model.config.n_layer  # For GPT-2, this is 12

# Initialize results storage
intervention_counts = np.zeros(num_layers)  # Counts of rule changes per layer
total_runs = 0  # Total number of runs

# Loop over multiple runs with different rules
num_prompts = 10  # Adjust based on your resources
for i in tqdm(range(num_prompts)):
    # Select a rule and corresponding labeling function
    rule_description = random.choice(rule_descriptions)

    # Generate object descriptions and labels
    object_descriptions = [
        '- sml yel cir',
        '- lrg blu tri',
        '- med grn rec',
        '- sml blu rec',
        '- lrg yel tri',
        '- med blu cir',
        '- sml grn rec',
    ]

    # Label examples using the rule
    labeled_examples = []
    for obj in object_descriptions[:5]:  # Use first 5 for examples
        label = label_object(obj, rule_description)
        labeled_examples.append(f"{obj} -> {label}")

    # Construct the prompt
    def generate_prompt():
        prompt = (
            "Learn the secret rule to label the objects correctly. "
            "If an object follows the rule, it should be labeled 'True'. "
            "Otherwise, it should be labeled 'False'. "
            "Always follow the arrow '->' with an object's label.\n\n"
            "Here are some objects and their labels:\n"
        )
        for example in labeled_examples:
            prompt += f"{example}\n"
        prompt += "\nLabel the following object:\n"
        test_object = object_descriptions[5]  # Use the 6th object as the test object
        prompt += f"{test_object}\n\n"
        prompt += (
            "Provide the label for the object above, then explain the rule you've inferred starting with "
            "'Based on your examples, I've learned that...'"
        )
        return prompt, test_object

    prompt, test_object = generate_prompt()
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    # Generate correct and incorrect runs
    def generate_output(seed, temperature):
        set_seed(seed)
        activations = {}

        # Function to save activations
        def save_activation(layer_name):
            def hook(module, input, output):
                activations[layer_name] = output.detach()
            return hook

        # Register hooks on all layers
        handles = []
        for layer_num in range(num_layers):
            layer = model.transformer.h[layer_num]
            handle = layer.register_forward_hook(save_activation(f'layer_{layer_num}'))
            handles.append(handle)

        # Generate output
        output_sequences = model.generate(
            input_ids=inputs['input_ids'],
            max_length=inputs['input_ids'].shape[1] + 100,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Remove hooks
        for handle in handles:
            handle.remove()

        generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        return generated_text, activations

    # Obtain correct run
    max_attempts = 5
    correct_run = None
    for attempt in range(max_attempts):
        seed = random.randint(0, 10000)
        output_correct, activations_correct = generate_output(seed, temperature=0.7)
        assigned_label = label_test_object_in_output(output_correct, test_object)
        if assigned_label == label_object(test_object, rule_description):
            correct_run = {
                'text': output_correct,
                'activations': activations_correct,
            }
            break

    # Obtain incorrect run
    incorrect_run = None
    for attempt in range(max_attempts):
        seed = random.randint(0, 10000)
        output_incorrect, activations_incorrect = generate_output(seed, temperature=1.0)
        assigned_label = label_test_object_in_output(output_incorrect, test_object)
        if assigned_label and assigned_label != label_object(test_object, rule_description):
            incorrect_run = {
                'text': output_incorrect,
                'activations': activations_incorrect,
            }
            break

    # Proceed only if both runs are found
    if not correct_run or not incorrect_run:
        continue

    # Identify the categorization token position (assumed to be the same in both runs)
    label_pos_correct, rule_pos_correct = find_positions(correct_run['text'])
    label_pos_incorrect, rule_pos_incorrect = find_positions(incorrect_run['text'])

    if label_pos_correct == 0 or label_pos_incorrect == 0:
        continue  # Skip if label positions can't be found

    # For each layer, perform the intervention
    for layer_num in range(num_layers):
        # Prepare the intervention configuration
        intervention_config = IntervenableConfig(
            representations=[
                RepresentationConfig(
                    layer=layer_num,
                    component="residual",
                ),
            ],
            intervention_types=VanillaIntervention,
        )

        # Initialize the IntervenableModel
        intervenable_model = IntervenableModel(intervention_config, model)

        # Prepare unit_locations for swapping activations at the categorization token position
        unit_locations = {
            "sources->base": (
                [[label_pos_incorrect]],  # Position in source
                [[label_pos_correct]],    # Position in base
            )
        }

        # Run the intervention
        with torch.no_grad():
            _, intervened_outputs = intervenable_model(
                base=inputs,
                sources=inputs,
                activations={
                    'base': correct_run['activations'],
                    'sources': incorrect_run['activations'],
                },
                unit_locations=unit_locations,
                max_length=inputs['input_ids'].shape[1] + 100,
                do_sample=False,  # Keep deterministic to isolate effect of intervention
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode the intervened output
        intervened_text = tokenizer.decode(intervened_outputs.sequences[0], skip_special_tokens=True)

        # Extract the rule from the intervened output
        rule_intervened = extract_rule(intervened_text)

        # Compare the intervened rule to the correct rule
        similarity_intervened = compare_rules(rule_intervened, rule_description)

        # If the intervened rule does not match the correct rule, count as a change
        if not similarity_intervened:
            intervention_counts[layer_num] += 1

    total_runs += 1

# After processing all prompts, calculate the proportion
intervention_proportions = intervention_counts / total_runs

# Plot the results
import matplotlib.pyplot as plt

layers = np.arange(num_layers)
plt.figure(figsize=(10, 6))
plt.bar(layers, intervention_proportions)
plt.xlabel('Layer')
plt.ylabel('Proportion of Rule Changes')
plt.title('Effect of Swapping Categorization Activations at Each Layer on Rule Elicitation')
plt.xticks(layers)
plt.show()
