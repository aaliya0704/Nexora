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
