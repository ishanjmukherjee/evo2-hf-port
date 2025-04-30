# # save this as inspect_keys_minimal.py
# import torch, safetensors.torch, pathlib, sys

# # --- Make sure the script can find your custom modules ---
# sys.path.append(str(pathlib.Path(__file__).parent.resolve()))
# # --------------------------------------------------------

# try:
#     from configuration_evo2 import Evo2Config
#     from modeling_evo2 import Evo2ForCausalLM
# except ImportError as e:
#     print(f"Import Error: {e}\nEnsure config/modeling files are present.", file=sys.stderr)
#     sys.exit(1)

# # --- Config and Paths ---
# ROOT = pathlib.Path(".")
# CKPT_PATH = ROOT / "model.safetensors"
# if not CKPT_PATH.exists():
#     print(f"Checkpoint not found: {CKPT_PATH}", file=sys.stderr)
#     # Still try to load HF keys if config exists
#     # sys.exit(1) # Removed exit to allow printing HF keys anyway

# # --- Get HF Model Keys ---
# hf_keys = set()
# try:
#     print("Loading HF model structure...")
#     config = Evo2Config.from_pretrained(ROOT)
#     # Try meta device init first for speed, fallback to CPU
#     try:
#         model = Evo2ForCausalLM._from_config(config, torch_dtype=torch.bfloat16)
#     except Exception:
#         print("Meta init failed, using CPU init...")
#         model = Evo2ForCausalLM(config)
#     hf_keys = set(model.state_dict().keys())
#     print("\n--- HF Model Keys ---")
#     for k in sorted(list(hf_keys)): print(k)
#     del model # Free memory
# except Exception as e:
#     print(f"\nError loading HF model structure: {e}", file=sys.stderr)

# # --- Get Checkpoint Keys ---
# ckpt_keys = set()
# if CKPT_PATH.exists():
#     try:
#         print("\nLoading checkpoint keys...")
#         ckpt_keys = set(safetensors.torch.load_file(CKPT_PATH, device="cpu").keys())
#         print("\n--- Checkpoint Keys ---")
#         for k in sorted(list(ckpt_keys)): print(k)
#     except Exception as e:
#         print(f"\nError loading checkpoint {CKPT_PATH}: {e}", file=sys.stderr)

# # --- Compare Keys ---
# if hf_keys and ckpt_keys:
#     print("\n--- Key Differences ---")
#     print(f"\nIn HF Model ONLY ({len(hf_keys - ckpt_keys)}):")
#     for k in sorted(list(hf_keys - ckpt_keys)): print(f"  {k}")
#     print(f"\nIn Checkpoint ONLY ({len(ckpt_keys - hf_keys)}):")
#     for k in sorted(list(ckpt_keys - hf_keys)): print(f"  {k}")

# print("\nDone.")

"""
-------------------
"""

import torch, safetensors.torch, pathlib, sys
root = pathlib.Path(".")

import torch, warnings, json, pathlib
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM

# root = pathlib.Path(".")
# print("Loading tokenizer…")
# tok   = AutoTokenizer.from_pretrained(root, trust_remote_code=True)

# print("Loading model… (this takes ~30 s on first run)")
# model = AutoModelForCausalLM.from_pretrained(
#             root,
#             torch_dtype="auto",          # uses bf16/fp16 if your GPU supports it
#             device_map="auto",           # spreads across multiple GPUs if present
#             trust_remote_code=True)
# hf_keys = set(model.state_dict().keys())
# print("\n--- HF Model Keys ---")
# for k in sorted(list(hf_keys)):
#     print(k)

ROOT = pathlib.Path(".")
CKPT_PATH = ROOT / "model.safetensors"

ckpt_keys = set()
if CKPT_PATH.exists():
    try:
        print("\nLoading checkpoint keys...")
        ckpt = safetensors.torch.load_file(CKPT_PATH, device="cpu")
        ckpt_keys = set(ckpt.keys())
        # print("\n--- Checkpoint Keys ---")
        # for k in sorted(list(ckpt_keys)):
        #     print(k)
        # print("\n--- End Checkpoint Keys ---")

        non_tensors = {}
        for k, v in ckpt.items():
            if not isinstance(v, torch.Tensor):
                non_tensors[k] = type(v)

        if non_tensors:
            print("\nWARNING: Found non-tensor objects in model.safetensors!")
            for key, obj_type in non_tensors.items():
                print(f"  Key: '{key}', Type: {obj_type}")
        else:
            print("\nAll objects in model.safetensors are Tensors.")
    except Exception as e:
        print(f"\nError loading checkpoint {CKPT_PATH}: {e}", file=sys.stderr)
