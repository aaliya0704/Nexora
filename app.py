from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    TextIteratorStreamer,
)
from threading import Thread
import torch


# -------------------------
# Model Configuration
# -------------------------

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
# -------------------------
# Generation Configuration
# -------------------------

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1
DO_SAMPLE = True

generation_config = GenerationConfig(
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    repetition_penalty=REPETITION_PENALTY,
    do_sample=DO_SAMPLE,
)

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
MAX_HISTORY = 20


def build_prompt():
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    messages.extend(conversation_history)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return prompt


# -------------------------
# Response Generator
# -------------------------


def generate_response(prompt):
    # Convert prompt into tokens
    inputs = tokenizer(prompt, return_tensors="pt")

    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    # Create the streamer
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    # Arguments for model.generate()
    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "generation_config": generation_config,
        "streamer": streamer,
    }

    # Run generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Collect the streamed text
    response = ""

    for new_text in streamer:
        print(new_text, end="", flush=True)
        response += new_text

    print()  # Move to the next line after streaming finishes

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

    if len(conversation_history) > MAX_HISTORY:
        conversation_history[:] = conversation_history[-MAX_HISTORY:]

    print(f"\nConversation History: {len(conversation_history)} messages")

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

        print("\nNexora: ", end="", flush=True)

        chat_response(user_input)


if __name__ == "__main__":
    main()
