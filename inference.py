import torch, warnings, json, pathlib
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from configuration_evo2 import Evo2Config

# --- Configuration ---
root = pathlib.Path(".")
# Choose a reasonable max sequence length for your inference task
# Needs to be >= prompt_length + max_new_tokens
INFERENCE_MAX_SEQLEN = 8192 # Or 4096, 16384, etc.

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(root, trust_remote_code=True)

print(f"Loading configuration and overriding max_seqlen to {INFERENCE_MAX_SEQLEN}...")
# Load the configuration object from the directory
config = Evo2Config.from_pretrained(root, trust_remote_code=True)
# Override the max_seqlen in the loaded config object
config.max_seqlen = INFERENCE_MAX_SEQLEN
# You might also want to ensure max_batch_size is appropriate if you change batching
# config.max_batch_size = YOUR_BATCH_SIZE # Defaults to 1 in your config

print("Loading model with modified config... (this takes ~30 s on first run)")
model = AutoModelForCausalLM.from_pretrained(
            root,
            config=config, # Pass the modified config object
            torch_dtype=torch.bfloat16, # Specify dtype explicitly
            device_map="cuda:0",       # Specify device map explicitly
            trust_remote_code=True
            # Add quantization here if needed, e.g., load_in_8bit=True
            )

# quick smoke-test
prompt = "ATGGCGA"           # 8-mer DNA seed
tokens = tok(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(
            input_ids=tokens['input_ids'],
            max_new_tokens=64,
            temperature=0.8,
            do_sample=True)
print("\n--- Generated sequence ---\n",
      tok.decode(out[0], skip_special_tokens=True))
