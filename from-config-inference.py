import torch, warnings, json, pathlib

from configuration_evo2 import Evo2Config
from modeling_evo2 import Evo2ForCausalLM

cfg = Evo2Config.from_original_config(
        "configs/evo2_7b.json",
        torch_dtype="bfloat16"   # you can override fields here
)
model = Evo2ForCausalLM(cfg)        # or Evo2ForCausalLM(cfg)
print("Initialized!")

# smoke-test
# ids = tokenizer("ACGT\n", return_tensors="pt").input_ids.to(model.device)
# out = model.generate(ids, max_new_tokens=32)
# print(tokenizer.decode(out[0], skip_special_tokens=True))
