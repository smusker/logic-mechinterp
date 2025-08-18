from openai import OpenAI
import re

api_input_openrouter = input("Enter openrouter key: ")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_input_openrouter,
)

# User asks first question
messages = [
    {"role": "system", "content": "You are a careful, stepwise reasoner. Think in <think> blocks, then give a short final answer."},
    {"role": "user", "content": "Find the LCM of 48 and 180, and explain briefly."}
]

resp = client.chat.completions.create(
    model="mistralai/magistral-medium-2506:thinking", #note this is not open-weights. Small is open weights, but replacing "medium" with "small" doesn't work when keeping the thinking flag. 
    messages=messages,
    max_tokens=600,
    extra_body={
        "include_reasoning": True,
        "reasoning": {"effort": "medium"}
    }
)

print("printing resp 1")
print(resp)

msg = resp.choices[0].message
reply = msg.content or ""
think_text = msg.reasoning or ""

print("\nprinting reply 1")
print(reply)

think_block = f"<think>\n{think_text}\n</think>" if think_text else None
print("\nthink block is", think_block)

# New user follow-up
new_question = {"role": "user", "content": "Using your earlier steps, what’s the GCD of 48 and 180?"}

# Build context: include new question, then re-inject thinking as supplemental context
next_messages = [
    {"role": "system", "content": "You are a careful, stepwise reasoner. Think in <think> blocks, then give a short final answer."},
    {"role": "user", "content": "Find the LCM of 48 and 180, and explain briefly."},
    {"role": "assistant", "content": reply},
    new_question
]

if think_block:
    # Insert the reasoning trace as supplemental assistant context
    next_messages.append({"role": "assistant", "content": think_block})
    print("\nappended think block")

resp2 = client.chat.completions.create(
    model="mistralai/magistral-medium-2506:thinking",
    messages=next_messages,
    max_tokens=300,
    extra_body={
        "include_reasoning": True,
        "reasoning": {"effort": "medium"}
    }
)

print("\nprinting response 2")
print(resp2.choices[0].message.content)
