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
from pyvene import create_gpt2  # Replace with appropriate function if using a different model
from transformers import set_seed
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
# Replace 'gpt2' with your capable model (ensure it has the same architecture assumptions)
config, tokenizer, model = create_gpt2(name='gpt2')
model.to(device)
model.eval()

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)

# Define multiple rules and labeling functions
rule_descriptions = [
    "An object is labeled 'True' if it is BLU or a REC.",
    "An object is labeled 'True' if it is GRN and S.",
    "An object is labeled 'True' if it is not a CIR.",
    "An object is labeled 'True' if it is YEL and L.",
    "An object is labeled 'True' if it is RED or a TRI.",
    "An object is labeled 'True' if it is BLU and not a SQR.",
    "An object is labeled 'True' if it is not GRN or is L.",
    "An object is labeled 'True' if it is S and not RED.",
    "An object is labeled 'True' if it is a CIR or a REC.",
    "An object is labeled 'True' if it is not YEL and not S.",
    # Add more rules as needed
]

# Abbreviations for sizes, colors, and shapes
sizes = ['S', 'M', 'L']       # Small, Medium, Large
colors = ['BLU', 'GRN', 'YEL', 'RED']  # Blue, Green, Yellow, Red
shapes = ['CIR', 'TRI', 'REC', 'SQR']  # Circle, Triangle, Rectangle, Square

def label_object(description, rule_desc):
    # Simplified parsing based on the rule description
    tokens = description.replace('-', '').split()  # Remove '-' and split
    size = tokens[0].split('=')[1]
    color = tokens[1].split('=')[1]
    shape = tokens[2].split('=')[1]

    if "BLU or a REC" in rule_desc:
        is_blu = 'BLU' == color
        is_rec = 'REC' == shape
        return 'True' if is_blu or is_rec else 'False'
    elif "GRN and S" in rule_desc:
        is_grn = 'GRN' == color
        is_s = 'S' == size
        return 'True' if is_grn and is_s else 'False'
    elif "not a CIR" in rule_desc:
        is_cir = 'CIR' == shape
        return 'False' if is_cir else 'True'
    elif "YEL and L" in rule_desc:
        is_yel = 'YEL' == color
        is_l = 'L' == size
        return 'True' if is_yel and is_l else 'False'
    elif "RED or a TRI" in rule_desc:
        is_red = 'RED' == color
        is_tri = 'TRI' == shape
        return 'True' if is_red or is_tri else 'False'
    elif "BLU and not a SQR" in rule_desc:
        is_blu = 'BLU' == color
        is_sqr = 'SQR' == shape
        return 'True' if is_blu and not is_sqr else 'False'
    elif "not GRN or is L" in rule_desc:
        is_grn = 'GRN' == color
        is_l = 'L' == size
        return 'True' if not is_grn or is_l else 'False'
    elif "S and not RED" in rule_desc:
        is_s = 'S' == size
        is_red = 'RED' == color
        return 'True' if is_s and not is_red else 'False'
    elif "a CIR or a REC" in rule_desc:
        is_cir = 'CIR' == shape
        is_rec = 'REC' == shape
        return 'True' if is_cir or is_rec else 'False'
    elif "not YEL and not S" in rule_desc:
        is_yel = 'YEL' == color
        is_s = 'S' == size
        return 'True' if not is_yel and not is_s else 'False'
    else:
        return 'False'

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
        return output_text[start_idx:]
    else:
        return ""

def compare_rules(rule1, rule2):
    """
    Placeholder function to compare two rule descriptions.
    Returns True if the rules are effectively the same, False otherwise.
    Implement this function using an LLM API or semantic similarity measure.
    """
    # Simplified comparison for demonstration
    return rule1.strip().lower() == rule2.strip().lower()

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
        "'Based on your examples, I've learned that...'\n"
    )
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    tokenized_prompt = tokenizer.tokenize(prompt)
    
    # Get token positions
    token_positions = {}
    
    # Find positions of labels ('True', 'False') in the examples
    label_tokens = ['True', 'False']
    label_indices = []
    current_pos = 0
    for example in examples[:5]:
        # Find the position of the label token
        label = example.split(' -> ')[1]
        # Build the search string
        search_string = f"-> {label}"
        # Tokenize the search string
        search_tokens = tokenizer.tokenize(search_string)
        # Find the index of the first occurrence starting from current_pos
        try:
            idx = tokenized_prompt.index(search_tokens[0], current_pos)
            label_idx = idx + len(search_tokens) - 1  # Position of the label token
            label_indices.append(label_idx)
            current_pos = label_idx + 1
        except ValueError:
            continue

    # Position of the test object's label (which we need to predict)
    test_object_label_pos = len(tokenized_prompt)  # Since the model will generate this

    # Positions of the test object's attributes
    test_object_tokens = tokenizer.tokenize(test_object_line)
    test_object_token_start = tokenized_prompt.index(test_object_tokens[0])
    # Positions of 'Size=', 'Color=', 'Shape=' in the test object
    size_token = 'Size='
    color_token = 'Color='
    shape_token = 'Shape='
    size_idx = tokenized_prompt.index(size_token, test_object_token_start)
    color_idx = tokenized_prompt.index(color_token, size_idx + 1)
    shape_idx = tokenized_prompt.index(shape_token, color_idx + 1)

    token_positions = {
        'example_label_positions': label_indices,      # Positions of the labels in the examples
        'test_object_size': size_idx,                  # Position of 'Size=' in test object
        'test_object_color': color_idx,                # Position of 'Color=' in test object
        'test_object_shape': shape_idx,                # Position of 'Shape=' in test object
        'label_token': len(tokenized_prompt),          # Position where the model will generate the label
        'rule_start': len(tokenized_prompt) + 1,       # Position where the model will start the rule explanation
    }
    
    return prompt, inputs, token_positions, test_object_line

# Total number of layers in the model
num_layers = model.config.n_layer

# Components to intervene on
components = ['residual', 'mlp_activation', 'attention_output']

# Initialize results storage
num_positions = 6  # Number of positions we're intervening on
intervention_counts = np.zeros((len(components), num_layers, num_positions))
total_runs = 0

# Parameters for the experiment
num_rules = len(rule_descriptions)
num_prompts_per_rule = 5     # Number of prompts per rule
max_attempts = 5             # Max attempts to find correct/incorrect runs per test object

# Main experimental loop
for rule_desc in tqdm(rule_descriptions, desc="Processing Rules"):
    for prompt_idx in range(num_prompts_per_rule):
        # Generate prompt and token positions
        prompt, inputs, token_positions, test_object_line = generate_prompt_and_get_token_positions(rule_desc)
        
        # Generate correct and incorrect runs
        correct_run = None
        incorrect_run = None

        # Attempt to get a correct run
        for attempt in range(max_attempts):
            seed = random.randint(0, 10000)
            set_seed(seed)
            activations_correct = {}
            # Register hooks on all layers and components
            handles = []
            for layer_num in range(num_layers):
                layer = model.transformer.h[layer_num]
                for component in components:
                    if component == 'residual':
                        handle = layer.register_forward_hook(
                            lambda module, inp, outp, layer_num=layer_num, comp=component: activations_correct.setdefault((comp, layer_num), outp.detach())
                        )
                    elif component == 'mlp_activation':
                        handle = layer.mlp.register_forward_hook(
                            lambda module, inp, outp, layer_num=layer_num, comp=component: activations_correct.setdefault((comp, layer_num), outp.detach())
                        )
                    elif component == 'attention_output':
                        handle = layer.attn.register_forward_hook(
                            lambda module, inp, outp, layer_num=layer_num, comp=component: activations_correct.setdefault((comp, layer_num), outp.detach())
                        )
                    handles.append(handle)
            # Generate output
            output_sequences = model.generate(
                input_ids=inputs['input_ids'],
                max_length=inputs['input_ids'].shape[1] + 50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
            # Remove hooks
            for handle in handles:
                handle.remove()
            output_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
            assigned_label = label_test_object_in_output(output_text, test_object_line)
            true_label = label_object(test_object_line, rule_desc)
            if assigned_label == true_label:
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
            # Register hooks on all layers and components
            handles = []
            for layer_num in range(num_layers):
                layer = model.transformer.h[layer_num]
                for component in components:
                    if component == 'residual':
                        handle = layer.register_forward_hook(
                            lambda module, inp, outp, layer_num=layer_num, comp=component: activations_incorrect.setdefault((comp, layer_num), outp.detach())
                        )
                    elif component == 'mlp_activation':
                        handle = layer.mlp.register_forward_hook(
                            lambda module, inp, outp, layer_num=layer_num, comp=component: activations_incorrect.setdefault((comp, layer_num), outp.detach())
                        )
                    elif component == 'attention_output':
                        handle = layer.attn.register_forward_hook(
                            lambda module, inp, outp, layer_num=layer_num, comp=component: activations_incorrect.setdefault((comp, layer_num), outp.detach())
                        )
                    handles.append(handle)
            # Generate output
            output_sequences = model.generate(
                input_ids=inputs['input_ids'],
                max_length=inputs['input_ids'].shape[1] + 50,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
            # Remove hooks
            for handle in handles:
                handle.remove()
            output_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
            assigned_label = label_test_object_in_output(output_text, test_object_line)
            true_label = label_object(test_object_line, rule_desc)
            if assigned_label and assigned_label != true_label:
                incorrect_run = {
                    'text': output_text,
                    'activations': activations_incorrect,
                }
                break

        # Proceed only if both runs are found
        if not correct_run or not incorrect_run:
            continue

        # Positions to intervene on
        positions = [
            *token_positions['example_label_positions'],  # Positions of example labels
            token_positions['test_object_size'],          # Test object size
            token_positions['test_object_color'],         # Test object color
            token_positions['test_object_shape'],         # Test object shape
            token_positions['label_token'],               # Position where model generates label
            token_positions['rule_start']                 # Position where model starts rule explanation
        ]
        position_names = [
            'Example_Label_1', 'Example_Label_2', 'Example_Label_3', 'Example_Label_4', 'Example_Label_5',
            'Test_Object_Size', 'Test_Object_Color', 'Test_Object_Shape', 'Label_Token', 'Rule_Start'
        ]
        num_positions = len(positions)

        # Update intervention_counts to match num_positions
        intervention_counts = np.zeros((len(components), num_layers, num_positions))

        # Perform interventions over components, layers, and positions
        for comp_idx, component in enumerate(components):
            for layer_num in range(num_layers):
                for pos_idx, token_pos in enumerate(positions):
                    # Create intervention configuration for the current component, layer, and position
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

                    # Collect activations for the component and layer
                    key = (component, layer_num)
                    base_activations = {key: correct_run['activations'][key]}
                    source_activations = {key: incorrect_run['activations'][key]}

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
                            max_length=inputs['input_ids'].shape[1] + 50,
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
                        intervention_counts[comp_idx, layer_num, pos_idx] += 1

        total_runs += 1  # Increment total runs

# After processing all prompts and runs, calculate the proportions
intervention_proportions = intervention_counts / total_runs

# Plot the results for each component
for comp_idx, component in enumerate(components):
    plt.figure(figsize=(15, 6))
    plt.imshow(intervention_proportions[comp_idx], aspect='auto', cmap='viridis')
    plt.colorbar(label='Proportion of Rule Changes')
    plt.xlabel('Token Position')
    plt.ylabel('Layer')
    plt.title(f'Effect of Swapping {component} Activations on Rule Elicitation')
    plt.xticks(ticks=range(num_positions), labels=position_names, rotation=45)
    plt.yticks(ticks=range(num_layers))
    plt.tight_layout()
    plt.show()
