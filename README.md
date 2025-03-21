Set up and run: 

start logic_env
salloc -J interact -p 3090-gcondo -N 1-1 -n 1 --time 01:00:00 --gpus 3 --cpus-per-task 2 --mem 80G
pip install -r requirements.txt
export OPENAI_API_KEY=...
pip install huggingface_hub
huggingface-cli login
enter token

# logic-mechinterp

Rule elicitation faithfulness experiment: Do rule elicitations and inferences come from the same underlying representation?

Objective:
Investigate whether rule elicitation and categorization made by a language model originate from the same underlying representations. Specifically, determine if altering the model's internal representations at various token positions affects both the categorization decision and the subsequent rule elicitation, thereby testing the faithfulness of the rule explanation.

Background:
Inspired by the activation patching method used in the Space Needle example from the Pyvene paper, we aim to explore whether similar interventions can reveal a causal relationship between the model's categorization decisions and its rule elicitations.

Experimental Design:

Multiple Rules Sequentially:

Rules Tested:
"An object is labeled 'True' if it is BLU or a REC."
"An object is labeled 'True' if it is not a CIR."
"An object is labeled 'True' if it is RED and not an SQR."
Focusing on one rule at a time ensures prompt standardization and consistent token positions across runs.
Standardized Prompts:

Fixed Structure: Prompts follow a consistent format with fixed positions for object descriptions and labels.
Abbreviations: Use single-token abbreviations for sizes (S, M, L), colors (BLU, GRN, YEL, RED), and shapes (CIR, TRI, REC, SQR) to maintain consistent tokenization.
Fixed Label Pattern: Use a predetermined label pattern (e.g., 'False', 'True', 'True', 'False', 'True', 'False') to keep label positions consistent across all prompts.
Consistent Instructions: The instructional text and prompt structure remain the same for all runs.
Collecting Multiple Runs:

Correct Run: Generate a run where the model gives a categorization and then a rule. 
Incorrect Run: Selectively noise input tokens corresponding to the original output categorization until the categorization flips. We now have a flipped categorization followed by a rule. 
Goal: Obtain contrasting runs (e.g., 3 each per rule) to create pairs for intervention.
Activation Patching (Intervention):

Intervention Points: We intervene on sliding windows of tokens and layers, starting after standard input tokens and continuing up to and including initial output tokens. 
Swapping Activations: Swap activations from incorrect runs into correct runs at specified layers and token positions.
Combining Runs: Pair each correct run with each incorrect run, increasing data points per analysis (e.g., 3 correct × 3 incorrect = 9 combinations per rule).
Analyzing the Results:

Assessment: After interventions, observe whether changes in internal representations at various positions lead to changes in both the categorization and the rule elicitation.
Co-Variation Analysis: For each intervention, calculate the proportion of times where a change in categorization co-occurs with a change in the rule explanation.
Aggregate Statistics: Record the total number of interventions resulting in categorization changes and the number where the rule also changed, calculating the percentage of co-variation.
Coherence statistics: We assess the extent to which categorizations and stated rules remain consistent with each other when changing under interventions. 

Visualization: Generate heatmaps for each component (residual, MLP activation, attention output) to visualize the proportion of co-variations across layers and token positions.
Key Design Decisions:

Sequential Testing of Multiple Rules:

Allows exploration of the model's behavior across different logical constructs.
Facilitates comparison of co-variation patterns between different rules.
Standardized Prompts and Token Positions:

Ensures consistency in the input, which is crucial for meaningful interventions and accurate mapping of token positions.
Intervening Across All Token Positions:

Provides a comprehensive analysis of how different parts of the input affect both the model's decision and its explanation.
Aligns with the methodology used in the Space Needle example, allowing us to uncover long-range dependencies and complex interactions within the model.
Collecting Multiple Runs for Data Robustness:

Increases the reliability of results by enhancing statistical significance.
Variability control through random seeds and temperature settings provides necessary diversity in model outputs without altering the prompt structure.

Note:

Change and Reason:

Removed device_map="auto" to keep the model on a single device. Using inputs_embeds with model.generate() is incompatible with sharded models, causing device mismatch errors.
Future Consideration:

Larger models may require multi-GPU support for memory and computation.
Alternate Multi-GPU Solutions:

Avoid inputs_embeds with model.generate(), using input_ids instead.
Ensure tensors are correctly placed across devices.
Use Accelerate, DeepSpeed, or FairScale for parallelism.
Apply gradient checkpointing to reduce memory use.
These changes enable multi-GPU use while maintaining intervention compatibility.

**********************************//////////////////////******************************************

High level logic model experiment: How is the model reasoning about the underlying rules?

In this experiment, we study whether a model’s internal reasoning aligns more closely with one of two “XOR” causal models. We use a fixed token-length prompt describing basic “objects” (color + shape), and label them “True” or “False” under the rule “yellow XOR circle.” We then generate pairs of “base” and “source” examples for two candidate causal models:
• Model A: Treats XOR as a single logical operation.
• Model B: Decomposes XOR into OR, AND, and NOT steps.

We apply Boundless DAS to these pairs, training a rotation + boundary-masking subspace so that intervening on the base’s activations with the source’s activations reproduces the source’s outcome. Afterward, we:

Visualize per-layer, per-token swap effects in a heatmap.
Compute a global Interchange Intervention Accuracy (IIA) score for each causal model.
By comparing heatmaps and IIA scores, we determine which causal model best explains the network’s internal logic.
