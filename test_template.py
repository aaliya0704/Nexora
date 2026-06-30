from transformers import AutoTokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

messages = [
    {"role": "system", "content": "You are Nexora, a helpful AI assistant."},
    {"role": "user", "content": "Hello"},
]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

print(prompt)
