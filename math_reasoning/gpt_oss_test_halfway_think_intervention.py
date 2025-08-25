#Nemotron

from openai import OpenAI

# Prompt for your OpenRouter API key
api_input_openrouter = input("Enter openrouter key: ")

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_input_openrouter,
)

# Use Nemotron 8B on OpenRouter
MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1" #"nvidia/llama-3.1-nemotron-nano-8b-v1"

system_message = "You are a stepwise adder. Show your reasoning, then give the final numeric answer."

# First user question
messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": "Add 45 to 19."}
]

# Run with reasoning enabled (if supported by the model/provider)
resp = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    max_tokens=600,
    extra_body={
        "include_reasoning": True,
        "reasoning": {"effort": "medium"}
    }
)

print("==== First response ====")
msg = resp.choices[0].message
reply = msg.content or ""
think_text = getattr(msg, "reasoning", "") or ""

print("Assistant final reply:", reply)
print("Assistant reasoning:", think_text)

# --- splice reasoning halfway ---
half = think_text[: len(think_text)//2]
intervened_think = f"<think>\n{half}\nFinal Answer: **Step-by-Step Reasoning:** **Add the units place (ones):** 5 plus 9 is "

# Inject into conversation history (no closing </think>)
"""
next_messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": "Add 45 to 19."},
    {"role": "assistant", "content": reply},
    {"role": "assistant", "content": intervened_think},  # deliberately incomplete
    {"role": "user", "content": "Add 12 to 24."}
]
"""

next_messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": "Add 45 to 19."},
    {"role": "assistant", "content": intervened_think}
]

print("printing next_messages")
print(next_messages)
print("done printing next_messages")

# Run again
resp2 = client.chat.completions.create(
    model=MODEL,
    messages=next_messages,
    max_tokens=400,
    extra_body={
        "include_reasoning": True,
        "reasoning": {"effort": "medium"}
    }
)

print("\n==== Second response after intervention ====")
print("printing resp2")
print(resp2)
print("done printing resp2")

msg2 = resp2.choices[0].message
print("Reasoning (continued):\n", getattr(msg2, "reasoning", ""))
print("Final Answer:\n", msg2.content)






exit()

#OSS


from openai import OpenAI

# Prompt for your OpenRouter API key
api_input_openrouter = input("Enter openrouter key: ")

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_input_openrouter,
)

system_message = "You are a stepwise adder. Show your reasoning, then give the final numeric answer."

# First user question
messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": "Add 45 to 19."}
]

# Run with reasoning enabled
resp = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=messages,
    max_tokens=600,
    extra_body={
        "include_reasoning": True,
        "reasoning": {"effort": "medium"}
    }
)

print("==== First response ====")
msg = resp.choices[0].message
reply = msg.content or ""
think_text = msg.reasoning or ""

print("Assistant final reply:", reply)
print("Assistant reasoning:", think_text)

# --- splice reasoning halfway ---
half = think_text[: len(think_text)//2]
intervened_think = f"<think>\n{half}\nWait let me practice, 1 plus 1 is "

# Inject into conversation history (no closing </think>)
"""
next_messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": "Add 45 to 19."},
    {"role": "assistant", "content": reply},
    {"role": "assistant", "content": intervened_think},  # deliberately incomplete
    {"role": "user", "content": "Add 12 to 24."}
]
"""

next_messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": "Add 45 to 19."},
    {"role": "assistant", "content": intervened_think}
]

print("printing next_messages")

print(next_messages)

print("done printing next_messages")

# Run again
resp2 = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=next_messages,
    max_tokens=400,
    extra_body={
        "include_reasoning": True,
        "reasoning": {"effort": "medium"}
    }
)

print("\n==== Second response after intervention ====")

print("printing resp2")
print(resp2)
print("done printing resp2")
msg2 = resp2.choices[0].message
print("Reasoning (continued):\n", msg2.reasoning)
print("Final Answer:\n", msg2.content)
