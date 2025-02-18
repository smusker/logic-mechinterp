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

Correct Runs: Generate multiple runs where the model correctly categorizes the test object and provides the correct rule explanation.
Incorrect Runs: Generate multiple runs where the model incorrectly categorizes the test object and provides an incorrect rule explanation.
Variability: Use random seeds and temperature adjustments to introduce variability in the model's outputs.
Goal: Obtain a sufficient number of correct and incorrect runs (e.g., 3 each per rule) to create multiple combinations for intervention.
Activation Patching (Intervention):

Intervention Points: Intervene across multiple layers and all token positions in the prompt and initial output tokens, including:
All tokens in the input prompt, to comprehensively assess the influence of each position, as well as initial output tokens (further output tokens are omitted due to variable length complications). 
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

**********************************//////////////////////******************************************

High level logic model experiment: How is the model reasoning about the underlying rules?

What's tricky here is that in the original data for getting a new object categorization without rule elicitation, prompts are of different lengths as data changes and context expands, and the rule is always changing. This makes it difficult to use DAS, which assumes constant token positions and the ability to hand-specify some candidate causal models (which would be rule specific). So I have an idea to do one rule at a time (one rule has specifiable candidate causal models), and with fixed length prompts. 

We take one rule. We generate a lot of fixed length prompts (maybe 20 thousand, sufficient for boundless DAS training) that are consistent with this rule, with the same number of examples each time (so token positions are constant), but we change the object characteristics and associated true / false labels. In order to keep the prompts the same length while changing out content, we make all variable names single token and introduce them at the beginning of the prompt (e.g "We will use rec for rectangle..."). We specify by hand a few different high level causal models for how the LLM is solving the task. Maybe these models differ in order of operations, or grouping or something. Maybe one of the operators in the rule is an xor, which can be represented in the causal model either as one unitary operation, or as composed out of or plus and. We run boundless DAS like the Alpaca example and see if we get high IIA matching with one of the causal models. We repeat this process for maybe 3 rules.

From the above we'll be able to say: we can verify that the rule elicitations are faithful to however the model is deriving the answer, and we can verify that on this sample of a few rules, this is the interpretable underlying causal model the LLM is using.
