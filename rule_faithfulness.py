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
import random

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
# Using GPT-2 as a placeholder; replace 'gpt2' with your capable model
config, tokenizer, model = create_gpt2(name='gpt2')
model.to(device)
model.eval()

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)

# 1. Define the rule
rule_description = "An object is labeled 'True' if it is blue or a rectangle."

# 2. Function to label objects according to the rule
def label_object(description):
    # Simplified parsing
    is_blue = 'blu' in description
    is_rectangle = 'rec' in description
    if is_blue or is_rectangle:
        return 'True'
    else:
        return 'False'

# 3. Generate object descriptions
object_descriptions = [
    '- sml yel cir',
    '- lrg blu tri',
    '- med grn rec',
    '- sml blu rec',
    '- lrg yel tri',
    '- med blu cir',
    '- sml grn rec',
]

# 3a. Generate labels for the examples
labeled_examples = []
for obj in object_descriptions[:5]:  # Use first 5 for examples
    label = label_object(obj)
    labeled_examples.append(f"{obj} -> {label}")

# 4. Construct the prompt
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

# 5. Generate multiple completions using temperature for variation
def generate_with_activations(model, inputs, seed, temperature=0.9):
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
        max_length=inputs['input_ids'].shape[1] + 100,  # Adjust as needed
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

# Generate multiple outputs
num_attempts = 5
outputs = []
seeds = random.sample(range(1000), num_attempts)
for seed in seeds:
    output_text, activations = generate_with_activations(model, inputs, seed)
    outputs.append({
        'seed': seed,
        'text': output_text,
        'activations': activations,
    })

# 6. Identify one run with correct categorization and one with incorrect categorization
def label_test_object_in_output(output_text):
    # Find the label assigned to the test object
    # We look for the test object string and the following '->' and label
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

correct_run = None
incorrect_run = None
test_object_true_label = label_object(test_object)
for output in outputs:
    assigned_label = label_test_object_in_output(output['text'])
    if assigned_label is None:
        continue
    if assigned_label == test_object_true_label:
        if correct_run is None:
            correct_run = output
    else:
        if incorrect_run is None:
            incorrect_run = output
    if correct_run and incorrect_run:
        break

if not correct_run or not incorrect_run:
    print("Could not find both a correct and incorrect run.")
else:
    print("=== Correct Output ===")
    print(correct_run['text'])
    print("\n=== Incorrect Output ===")
    print(incorrect_run['text'])

    # 7. Prepare the intervention

    # Identify positions for swapping
    # Assuming that the label for the test object is right after the test object and '->'
    def find_positions(output_text):
        # Tokenize the output
        output_tokens = tokenizer.tokenize(output_text)
        # Find position of test object's last token
        test_object_tokens = tokenizer.tokenize(test_object.strip())
        try:
            start_idx = [i for i, x in enumerate(output_tokens) if x == test_object_tokens[-1]][0]
        except IndexError:
            start_idx = 0  # If not found, default to 0
        # Find '->' after test object
        try:
            arrow_idx = output_tokens.index('Ġ->', start_idx + 1)
        except ValueError:
            arrow_idx = start_idx
        # Label position is right after '->'
        label_pos = arrow_idx + 1 if arrow_idx + 1 < len(output_tokens) else arrow_idx
        # Find 'Based' starting the rule explanation
        try:
            rule_start_idx = output_tokens.index('ĠBased')
        except ValueError:
            rule_start_idx = len(output_tokens) - 1  # If not found, set to end

        # Adjust positions to account for prompt length
        prompt_length = inputs['input_ids'].shape[1]
        label_position = label_pos + prompt_length
        rule_position = rule_start_idx + prompt_length

        return label_position, rule_position

    label_pos_correct, rule_pos_correct = find_positions(correct_run['text'])
    label_pos_incorrect, rule_pos_incorrect = find_positions(incorrect_run['text'])

    # Ensure label positions are valid
    if label_pos_correct == 0 or label_pos_incorrect == 0:
        print("Could not find label positions properly.")
    else:
        # 8. Prepare and perform the intervention

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
                    'base': correct_run['activations'],
                    'sources': incorrect_run['activations'],
                },
                unit_locations=unit_locations,
                max_length=inputs['input_ids'].shape[1] + 100,
                do_sample=True,
                temperature=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode the intervened output
        intervened_text = tokenizer.decode(intervened_outputs.sequences[0], skip_special_tokens=True)

        print("\n=== Intervened Output ===")
        print(intervened_text)

        # 9. Compare the stated rules using an LLM API
        # We'll define a placeholder function for this

        def compare_rules(rule1, rule2):
            """
            Placeholder function to compare two rule descriptions.
            Returns True if the rules are effectively the same, False otherwise.
            """
            # In practice, implement this function using an LLM API
            # For example, use OpenAI's GPT-3 or ChatGPT to compare the semantic similarity
            # Since we cannot use the API here, we will simply return a dummy value
            return rule1.strip().lower() == rule2.strip().lower()

        # Extract the rules
        def extract_rule(output_text):
            start_phrase = "Based on your examples, I've learned that"
            start_idx = output_text.find(start_phrase)
            if start_idx != -1:
                return output_text[start_idx:]
            else:
                return ""

        rule_correct = extract_rule(correct_run['text'])
        rule_incorrect = extract_rule(incorrect_run['text'])
        rule_intervened = extract_rule(intervened_text)

        # Compare the rules to the true rule
        print("\n=== Rules Comparison ===")
        print("\nTrue Rule:")
        print(rule_description)
        print("\nRule in Correct Output:")
        print(rule_correct)
        print("\nRule in Incorrect Output:")
        print(rule_incorrect)
        print("\nRule in Intervened Output:")
        print(rule_intervened)

        # Compare rules
        similarity_correct = compare_rules(rule_correct, rule_description)
        similarity_incorrect = compare_rules(rule_incorrect, rule_description)
        similarity_intervened = compare_rules(rule_intervened, rule_description)

        print("\nComparison Results:")
        print(f"Rule in Correct Output matches True Rule? {similarity_correct}")
        print(f"Rule in Incorrect Output matches True Rule? {similarity_incorrect}")
        print(f"Rule in Intervened Output matches True Rule? {similarity_intervened}")

        # Determine if the rule meaningfully changed after intervention
        if similarity_intervened == similarity_incorrect:
            print("\nThe rule elicitation changed after swapping the categorization activations.")
        else:
            print("\nThe rule elicitation did not change significantly after swapping the categorization activations.")
else:
    print("Could not proceed with the intervention due to insufficient runs.")
