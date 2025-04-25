import torch
import safetensors.torch
import argparse
import pathlib
from collections import OrderedDict
import logging
import io # Import the io module

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def rename_key(key: str) -> str | None:
    """
    Renames keys from the original .pt checkpoint to match the
    Hugging Face model structure defined in modeling_evo2.py.

    Args:
        key: The original key from the .pt file.

    Returns:
        The renamed key, or None if the key should be skipped.
    """
    # Skip FP8 metadata keys - These often contain non-tensor data like BytesIO
    # which safetensors can't store directly and aren't usually needed for inference.
    if "_extra_state" in key:
        logging.debug(f"Skipping FP8 metadata key: {key}")
        return None
    # Skip dynamically computed 't' buffer if present
    if "filter.t" in key:
        logging.debug(f"Skipping dynamic buffer key: {key}")
        return None

    # Handle embedding layer
    if key == "embedding_layer.weight":
        # Target key in modeling_evo2.py: backbone.embedding_layer.word_embeddings.weight
        return "backbone.embedding_layer.word_embeddings.weight"

    # Handle final norm
    if key == "norm.scale":
        # Target key in modeling_evo2.py: backbone.norm.scale
        return "backbone.norm.scale"

    # Handle blocks (add backbone prefix)
    if key.startswith("blocks."):
        # Standard block structure: backbone.blocks.N.layer_name.parameter_name
        parts = key.split('.')
        # Example: blocks.0.pre_norm.scale -> backbone.blocks.0.pre_norm.scale
        # Example: blocks.3.inner_mha_cls.Wqkv.weight -> backbone.blocks.3.inner_mha_cls.Wqkv.weight
        # Example: blocks.0.filter.short_filter_weight -> backbone.blocks.0.filter.short_filter_weight
        # Example: blocks.0.mlp.l1.weight -> backbone.blocks.0.mlp.l1.weight
        # Example: blocks.0.projections.weight -> backbone.blocks.0.projections.weight (TELinear)
        # Example: blocks.0.out_filter_dense.weight -> backbone.blocks.0.out_filter_dense.weight

        # Skip potentially problematic keys from the provided list that don't seem
        # to map directly or might be TE internals we don't want.
        # 'mixer' keys seem specific to an older structure or debugging?
        if "mixer.attn" in key or "mixer.dense" in key or "mixer.mixer" in key:
             logging.warning(f"Skipping potentially problematic 'mixer' key: {key}")
             return None

        # The general renaming rule adds 'backbone.' prefix
        return f"backbone.{key}"

    # Skip unembedding weights if tied (handled by HF post_init)
    # If embeddings are not tied, this key might need a different target name.
    if key == "unembed.weight":
        logging.warning(f"Skipping potentially tied unembedding weight: {key}. "
                        "Ensure 'tie_word_embeddings=True' in HF config or handle manually.")
        return None

    logging.warning(f"Unhandled key: {key}. Skipping.")
    return None


def convert_pt_to_safetensors(pt_path: pathlib.Path, sf_path: pathlib.Path):
    """
    Loads weights from a .pt file, renames keys for HF compatibility,
    and saves them to a .safetensors file.

    Args:
        pt_path: Path to the input .pt checkpoint file.
        sf_path: Path to the output .safetensors file.
    """
    logging.info(f"Loading state dict from: {pt_path}")

    # *** Add io.BytesIO to the list of safe globals ***
    # This allows loading checkpoints that might contain FP8 metadata
    # stored as BytesIO objects, even with weights_only=True.
    torch.serialization.add_safe_globals([io.BytesIO])
    logging.info("Added io.BytesIO to safe globals for torch.load.")

    try:
        # Load the state dictionary onto the CPU
        # weights_only=True is safer and recommended when possible
        state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
        logging.info(f"Successfully loaded state dict with {len(state_dict)} keys using weights_only=True.")
    except Exception as e:
        logging.error(f"Failed to load with weights_only=True: {e}")
        logging.warning("Attempting to load with weights_only=False. "
                        "Ensure you trust the source of this checkpoint as it may execute arbitrary code.")
        try:
            # Fallback if weights_only=True fails even after adding BytesIO (less likely but possible)
            state_dict = torch.load(pt_path, map_location="cpu", weights_only=False)
            logging.info(f"Successfully loaded state dict with {len(state_dict)} keys using weights_only=False.")
        except Exception as final_e:
            logging.error(f"Failed to load state dict even with weights_only=False: {final_e}")
            return # Exit if loading fails completely

    # Create a new ordered dictionary for the renamed keys
    new_state_dict = OrderedDict()
    original_keys = list(state_dict.keys()) # Create a copy for iteration

    logging.info("Processing and renaming keys...")
    skipped_count = 0
    processed_count = 0
    for old_key in original_keys:
        new_key = rename_key(old_key)
        if new_key:
            # Ensure the tensor exists and is actually a tensor before adding
            tensor = state_dict[old_key]
            if isinstance(tensor, torch.Tensor):
                 new_state_dict[new_key] = tensor
                 processed_count += 1
                 logging.debug(f"Renamed '{old_key}' -> '{new_key}'")
            else:
                # This handles cases where weights_only=False might load non-tensor objects
                # that weren't skipped by name (e.g., if _extra_state contained something else)
                skipped_count += 1
                logging.warning(f"Skipped key '{old_key}' because its value is not a tensor (type: {type(tensor)}).")
        else:
            skipped_count += 1
            # Logging for skipped keys already happens in rename_key

    logging.info(f"Processed {processed_count} tensor keys, skipped {skipped_count} keys/non-tensors.")

    if not new_state_dict:
        logging.error("The resulting state dictionary is empty. Check key renaming logic and input file content.")
        return

    # Ensure the output directory exists
    sf_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Saving processed state dict to: {sf_path}")
    # Save the processed state dictionary
    # metadata can be added if needed, e.g., {"format": "pt"}
    try:
        safetensors.torch.save_file(new_state_dict, sf_path, metadata=None)
        logging.info("Successfully saved safetensors file.")
    except Exception as e:
        logging.error(f"Failed to save safetensors file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch .pt checkpoints to .safetensors format.")
    parser.add_argument("input_pt", type=str, help="Path to the input .pt checkpoint file.")
    parser.add_argument("output_safetensors", type=str, help="Path to the output .safetensors file.")
    args = parser.parse_args()

    input_path = pathlib.Path(args.input_pt)
    output_path = pathlib.Path(args.output_safetensors)

    if not input_path.is_file():
        logging.error(f"Input file not found: {input_path}")
    else:
        convert_pt_to_safetensors(input_path, output_path)
