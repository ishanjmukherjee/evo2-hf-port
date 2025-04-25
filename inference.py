import torch, warnings, json, pathlib
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM

root = pathlib.Path(".")
print("Loading tokenizer…")
tok   = AutoTokenizer.from_pretrained(root, trust_remote_code=True)

print("Loading model… (this takes ~30 s on first run)")
model = AutoModelForCausalLM.from_pretrained(
            root,
            torch_dtype="auto",          # uses bf16/fp16 if your GPU supports it
            device_map="auto",           # spreads across multiple GPUs if present
            trust_remote_code=True)

# quick smoke-test
prompt = "ATGGCGA"           # 8-mer DNA seed
tokens = tok(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(
            **tokens,
            max_new_tokens=64,
            temperature=0.8,
            do_sample=True)
print("\n--- Generated sequence ---\n",
      tok.decode(out[0], skip_special_tokens=True))
