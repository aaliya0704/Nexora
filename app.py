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

SYSTEM_PROMPT = (
    "You are Nexora, a helpful, friendly, and intelligent AI assistant. "
    "Answer clearly and accurately."
)


def build_prompt(user_input):
    prompt = (
        f"<|system|>\n{SYSTEM_PROMPT}</s>\n<|user|>\n{user_input}</s>\n<|assistant|>\n"
    )

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
    prompt = build_prompt(user_input)

    response = generate_response(prompt)

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
