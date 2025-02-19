import openai

def compare_rules(rule1, rule2):
    """
    Uses OpenAI's GPT-4 API to determine if two rules are equivalent in meaning.
    """
    prompt = (
        "Here are a few examples of equivalent rules:\n"
        "- 'An object is labeled True if it is blue or a rectangle'\n"
        "- 'An object is labeled True if it is a rectangle or blue'\n"
        "- 'All objects are True if they are a rectangle or if they are blue, False otherwise'\n"
        "- 'An object is labeled True if it is BLU or a REC'\n"
        "- 'An object is labeled True if it is a REC or BLU'\n"
        "- 'An object is True if it is BLU or if it is a REC, False otherwise'\n\n"
        "Now respond about these two rules.\n\n"
        f"Are the following two rules logically equivalent in meaning?\n\n"
        f"Rule 1: {rule1}\n"
        f"Rule 2: {rule2}\n\n"
        "Please answer 'True' if they are equivalent and 'False' if they are not."
    )
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a logical reasoning assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.0  # Make the response deterministic
    )
    
    answer = response["choices"][0]["message"]["content"].strip().lower()
    return answer == "true"

def assess_rule_consistency(assigned_label, test_object_description, rule_description):
    """
    Uses OpenAI's GPT-4 API to determine if a labeled object matches the given rule.
    """
    prompt = (
        "Here are examples of applying rules to objects:\n"
        "- Rule: 'An object is labeled True if it is blue or a rectangle.'\n"
        "  Object: 'A BLU TRI' -> Labeled True (Correct)\n"
        "  Object: 'A RED CIR' -> Labeled False (Correct)\n"
        "- Rule: 'An object is labeled True if it is not a circle.'\n"
        "  Object: 'A REC' -> Labeled True (Correct)\n"
        "  Object: 'A CIR' -> Labeled False (Correct)\n"
        "- Rule: 'An object is labeled True if it is RED and not an SQR.'\n"
        "  Object: 'A RED TRI' -> Labeled True (Correct)\n"
        "  Object: 'A RED SQR' -> Labeled False (Correct)\n\n"
        "Now evaluate the following:\n"
        f"Rule: {rule_description}\n"
        f"Object: {test_object_description}\n"
        f"Assigned Label: {assigned_label}\n\n"
        "Does the assigned label correctly follow the rule?\n"
        "Please answer 'True' if the label is correct and 'False' if it is incorrect."
    )
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a logical reasoning assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.0  # Make the response deterministic
    )
    
    answer = response["choices"][0]["message"]["content"].strip().lower()
    return answer == "true"
