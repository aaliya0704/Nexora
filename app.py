from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# -------------------------
# Model Configuration
# -------------------------

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype="auto", device_map="auto"
)

print("✅ Nexora is ready!")

# -------------------------
# Prompt Builder
# -------------------------

SYSTEM_PROMPT = """
You are Nexora, a helpful, friendly, and intelligent AI assistant.

Your job is to answer the user's questions clearly, accurately, and honestly.

Guidelines:
- Respond naturally like a conversational assistant.
- Answer only what the user asks.
- Do not invent previous conversations or user information.
- If you do not know something, admit it instead of making up an answer.
- Keep responses concise unless the user asks for more detail.
- Be polite, professional, and easy to understand.
- Never pretend the user said something they didn't.
"""


# -------------------------
# Conversation History
# -------------------------

conversation_history = []


def build_prompt():
    prompt = f"<|system|>\n{SYSTEM_PROMPT}</s>\n"

    for message in conversation_history:
        if message["role"] == "user":
            prompt += f"<|user|>\n{message['content']}</s>\n"

        elif message["role"] == "assistant":
            prompt += f"<|assistant|>\n{message['content']}</s>\n"

    prompt += "<|assistant|>\n"

    return prompt


# -------------------------
# Response Generator
# -------------------------


def generate_response(prompt):
    # Convert prompt into tokens
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate output tokens
    output = model.generate(
        **inputs, max_new_tokens=100, temperature=0.7, do_sample=True
    )

    # Remove the original prompt tokens
    generated_tokens = output[0][inputs["input_ids"].shape[1] :]

    # Decode only the newly generated tokens
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return response.strip()


# -------------------------
# Chat Function
# -------------------------


def chat_response(user_input):
    # Store the user's message
    conversation_history.append({"role": "user", "content": user_input})

    prompt = build_prompt()

    response = generate_response(prompt)

    # Store Nexora's reply
    conversation_history.append({"role": "assistant", "content": response})

    return response


# -------------------------
# Terminal Chat Interface
# -------------------------


def main():
    print("=" * 50)
    print("🤖 Welcome to Nexora AI!")
    print("Type 'exit' to end the conversation.")
    print("=" * 50)

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() in ["exit", "quit"]:
            print("\nNexora: Goodbye! Have a great day.")
            break

        response = chat_response(user_input)

        print(f"\nNexora: {response}")


if __name__ == "__main__":
    main()
