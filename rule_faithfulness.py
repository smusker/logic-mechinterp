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
# Replace with appropriate imports if using a different model
from transformers import GPT2Tokenizer, GPT2LMHeadModel, set_seed
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
# Replace 'gpt2' with a capable model that suits your needs
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token exists
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.to(device)
model.eval()

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)

# Abbreviations for sizes, colors, and shapes
sizes = ['S', 'M', 'L']       # Small, Medium, Large
colors = ['BLU', 'GRN', 'YEL', 'RED']  # Blue, Green, Yellow, Red
shapes = ['CIR', 'TRI', 'REC', 'SQR']  # Circle, Triangle, Rectangle, Square

# Define a single rule and its labeling function
rule_description = "An object is labeled 'True' if it is BLU or a REC."

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
    label_positions = []
    current_pos = 0
    for example in examples[:5]:
        label = example.split(' -> ')[1]
        search_string = f"-> {label}"
        search_tokens = tokenizer.tokenize(search_string)
        try:
            idx = tokenized_prompt.index(search_tokens[0], current_pos)
            label_idx = idx + len(search_tokens) - 1  # Position of the label token
            label_positions.append(label_idx)
            current_pos = label_idx + 1
        except ValueError:
            continue

    # Position of the test object's label (to be generated)
    label_token_position = len(tokenized_prompt)

    # Positions of the test object's attributes
    test_object_tokens = tokenizer.tokenize(test_object_line)
    test_object_token_start = tokenized_prompt.index(test_object_tokens[0])
    # Positions of 'Size=', 'Color=', 'Shape=' in the test object
    size_token = 'Size='
    color_token = 'Color='
    shape_token = 'Shape='
    size_idx = tokenized_prompt.index(size_token, test_object_token_start)
    color_idx = tokenizer.tokenize('=')[0]
    color_idx = size_idx + 2  # Assuming fixed positions
    shape_idx = size_idx + 4  # Assuming fixed positions

    token_positions = {
        'example_label_positions': label_positions,      # Positions of labels in examples
        'test_object_size': size_idx + 1,                # Position of size value in test object
        'test_object_color': size_idx + 3,               # Position of color value in test object
        'test_object_shape': size_idx + 5,               # Position of shape value in test object
        'label_token': label_token_position,             # Position where the model will generate the label
        'rule_start': label_token_position + 1,          # Position where the rule explanation starts
    }
    
    return prompt, inputs, token_positions, test_object_line

# Total number of layers in the model
num_layers = model.config.n_layer

# Components to intervene on
components = ['residual', 'mlp_activation', 'attention_output']

# Positions to intervene on (we'll determine these after tokenization)
position_names = [
    'Example_Label_1', 'Example_Label_2', 'Example_Label_3', 'Example_Label_4', 'Example_Label_5',
    'Test_Object_Size', 'Test_Object_Color', 'Test_Object_Shape', 'Label_Token', 'Rule_Start'
]

num_positions = len(position_names)

# Initialize results storage
intervention_counts = np.zeros((len(components), num_layers, num_positions))
total_interventions = np.zeros((len(components), num_layers, num_positions))

# Number of correct and incorrect runs to collect
num_correct_runs = 3
num_incorrect_runs = 3

# Collect correct and incorrect runs
correct_runs = []
incorrect_runs = []

prompt, inputs, token_positions, test_object_line = generate_prompt_and_get_token_positions(rule_description)

print("Prompt:")
print(prompt)

# List to keep track of seeds used
used_seeds = set()

# Function to capture activations with correct variable scoping
def get_activation_capturer(activations_dict, key):
    def capturer(module, input, output):
        activations_dict[key] = output.detach()
    return capturer

# Collect correct runs
attempts = 0
while len(correct_runs) < num_correct_runs and attempts < 50:
    seed = random.randint(0, 10000)
    if seed in used_seeds:
        continue
    used_seeds.add(seed)
    set_seed(seed)
    activations = {}
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
    true_label = label_object(test_object_line, rule_description)
    if assigned_label == true_label:
        correct_runs.append({
            'text': output_text,
            'activations': activations,
        })
    attempts += 1

# Collect incorrect runs
attempts = 0
while len(incorrect_runs) < num_incorrect_runs and attempts < 50:
    seed = random.randint(0, 10000)
    if seed in used_seeds:
        continue
    used_seeds.add(seed)
    set_seed(seed)
    activations = {}
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
    true_label = label_object(test_object_line, rule_description)
    if assigned_label is not None and assigned_label != true_label:
        incorrect_runs.append({
            'text': output_text,
            'activations': activations,
        })
    attempts += 1

# Ensure we have enough runs
if len(correct_runs) < num_correct_runs or len(incorrect_runs) < num_incorrect_runs:
    print("Not enough correct and incorrect runs were collected.")
else:
    # Create combinations of correct and incorrect runs
    total_runs = 0
    for correct_run in correct_runs:
        for incorrect_run in incorrect_runs:
            # Perform interventions over components, layers, and positions
            for comp_idx, component in enumerate(components):
                for layer_num in range(num_layers):
                    for pos_idx, position_name in enumerate(position_names):
                        # Positions to intervene on
                        if position_name.startswith('Example_Label'):
                            # Get the index from the position name
                            label_idx = int(position_name.split('_')[-1]) - 1
                            token_pos = token_positions['example_label_positions'][label_idx]
                        else:
                            token_pos = token_positions[position_name.lower()]
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
                        rule_correct = extract_rule(correct_run['text'])
                        rule_incorrect = extract_rule(incorrect_run['text'])

                        # Compare intervened rule to the correct rule
                        similarity_intervened = compare_rules(rule_intervened, rule_description)

                        # Record the result
                        total_interventions[comp_idx, layer_num, pos_idx] += 1
                        if not similarity_intervened:
                            intervention_counts[comp_idx, layer_num, pos_idx] += 1
            total_runs += 1  # Count each intervention combination

    # Calculate the proportions
    intervention_proportions = np.divide(intervention_counts, total_interventions, out=np.zeros_like(intervention_counts), where=total_interventions!=0)

    # Plot the results for each component
    for comp_idx, component in enumerate(components):
        plt.figure(figsize=(15, 6))
        plt.imshow(intervention_proportions[comp_idx], aspect='auto', cmap='viridis')
        plt.colorbar(label='Proportion of Rule Changes')
        plt.xlabel('Token Position')
        plt.ylabel('Layer')
        plt.title(f'Effect of Swapping {component} Activations on Rule Elicitation')
        plt.xticks(ticks=range(num_positions), labels=position_names, rotation=45, ha='right')
        plt.yticks(ticks=range(num_layers))
        plt.tight_layout()
        plt.show()
