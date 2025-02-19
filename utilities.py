import openai

def compare_rules(rule1, rule2):
    """
    Uses OpenAI's GPT-4 API to determine if two rules are equivalent in meaning.
    """
    prompt = (
        "Here are a few examples of equivalent rules:\n"
        "- 'An object is labeled True if it is blue or a rectangle'\n"
        "- 'An object is labeled True if it is a rectangle or blue'\n"
        "- 'All objects are True if they are a rectangle or if they are blue, False otherwise'\n\n"
        "Now respond about these two rules."
        f"Are the following two rules logically equivalent in meaning?\n\n"
        f"Rule 1: {rule1}\n"
        f"Rule 2: {rule2}\n\n"
        "Please answer 'True' if they are equivalent and 'False' if they are not."
    )
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful assistant skilled in logical reasoning."},
                  {"role": "user", "content": prompt}],
        temperature=0.0  # Make the response deterministic
    )
    
    answer = response["choices"][0]["message"]["content"].strip().lower()
    return answer == "true"
