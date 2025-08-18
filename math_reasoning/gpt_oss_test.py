from openai import OpenAI
import re

# Prompt for your OpenRouter API key
api_input_openrouter = input("Enter openrouter key: ")

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_input_openrouter,
)

system_message = "You are a stepwise adder. First add the units with carry, then add the next significant digits one at a time, then report the result. When responding to the user, give only the final answer."

# First user question
messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": "Add 31 to 17."}
]

# Call GPT-OSS-20B with reasoning enabled
resp = client.chat.completions.create(
    model="openai/gpt-oss-20b",   # ✅ open-weight reasoning model
    messages=messages,
    max_tokens=600,
    extra_body={
        "include_reasoning": True,        # return reasoning in the response
        "reasoning": {"effort": "medium"} # low | medium | high
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
new_question = {"role": "user", "content": "Add 12 to 24."}

# Build conversation history
next_messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": "Add 31 to 17."},
    {"role": "assistant", "content": reply},
    new_question
]

if think_block:
    # Add the reasoning trace as context
    next_messages.append({"role": "assistant", "content": think_block})
    print("\nappended think block")

# Follow-up question with reasoning enabled
resp2 = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=next_messages,
    max_tokens=300,
    extra_body={
        "include_reasoning": True,
        "reasoning": {"effort": "medium"}
    }
)

print("\nprinting response 2")
msg2 = resp2.choices[0].message
print("Reasoning:\n", msg2.reasoning)
print("Final Answer:\n", msg2.content)
