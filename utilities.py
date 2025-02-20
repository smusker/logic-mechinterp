import openai
import time

def exponential_backoff(api_call, max_retries=5, base_delay=1):
    retries = 0
    while retries < max_retries:
        try:
            return api_call()
        except openai.error.RateLimitError:
            delay = base_delay * (2 ** retries)
            time.sleep(delay)
            retries += 1
    raise Exception("Max retries exceeded for API call")

def api_call(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a logical reasoning assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.0
    )
    return response["choices"][0]["message"]["content"].strip().lower()

def compare_rules(rule1, rule2):
    """
    Uses OpenAI's GPT-4o mini API to determine if two rules are equivalent in meaning.
    """
    if rule1.strip().lower() == rule2.strip().lower():
        return True
    
    prompt = (
        "We will abbreviate sizes as: S=small, M=medium, L=large; "
        "colors as: BLU=blue, GRN=green, YEL=yellow, RED=red; "
        "shapes as: CIR=circle, TRI=triangle, REC=rectangle, SQR=square.\n\n"
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
        "Please answer 'True' if they are equivalent and 'False' if they are not. Respond now with only the word true or false: "
    )
    
    answer = exponential_backoff(lambda: api_call(prompt))
    return "true" in answer

def assess_rule_consistency(assigned_label, test_object_description, rule_description):
    """
    Uses OpenAI's GPT-4o mini API to determine if a labeled object matches the given rule.
    """
    prompt = (
        "We will abbreviate sizes as: S=small, M=medium, L=large; "
        "colors as: BLU=blue, GRN=green, YEL=yellow, RED=red; "
        "shapes as: CIR=circle, TRI=triangle, REC=rectangle, SQR=square.\n\n"
        "Here are examples of applying rules to objects:\n\n"
        "- Rule: 'An object is labeled True if it is blue or a rectangle.'\n"
        "  Object: 'A BLU TRI' -> Labeled False (consistency: false)\n"
        "  Object: 'A RED CIR' -> Labeled False (consistency: true)\n\n"
        "- New rule: 'An object is labeled True if it is not a circle.'\n"
        "  Object: 'A REC' -> Labeled True (consistency: true)\n"
        "  Object: 'A CIR' -> Labeled True (consistency: false)\n\n"
        "- New rule: 'An object is labeled True if it is RED and not an SQR.'\n"
        "  Object: 'A RED TRI' -> Labeled True (consistency: true)\n"
        "  Object: 'A RED SQR' -> Labeled False (consistency: true)\n\n"
        "Now evaluate the following:\n"
        f"New rule: {rule_description}\n"
        f"Object: {test_object_description}\n"
        f"Assigned Label: {assigned_label}\n\n"
        "Does the assigned label consistently follow the rule?\n"
        "Please answer 'True' if the label is correct and 'False' if it is incorrect. Respond now with only the word true or false: "
    )
    
    answer = exponential_backoff(lambda: api_call(prompt))
    return "true" in answer
