"""New file, authored myself: Hugging Face configuration file for StripedHyena-2
(Evo 2) models. Based on Together's configuration_hyena.py but extended to cover
all Evo 2-specific hyper-parameters from the provided config.json.
"""

from typing import List, Optional
import json
from transformers.configuration_utils import PretrainedConfig


class Evo2Config(PretrainedConfig):
    """Configuration class for Evo 2 (StripedHyena-2) causal-LM checkpoints.

    Every keyword argument listed here has the same default value as in the
    reference `config.json`.  Additional keys coming from the Hugging Face
    `PretrainedConfig` base (e.g. *bos_eos_token_id*) can still be supplied via
    **kwargs.
    """

    model_type: str = "evo2"

    def __init__(
        self,
        # === Vocabulary & embedding ===
        vocab_size: int = 512,
        hidden_size: int = 4096,
        tie_embeddings: bool = True,
        make_vocab_size_divisible_by: int = 8,
        # === Depth / width ===
        num_hidden_layers: int = 32,
        num_layers: int = 32,
        num_attention_heads: int = 32,
        # === Filter dimensions ===
        num_filters: int = 4096,
        inner_mlp_size: int = 11264,
        inner_size_multiple_of: int = 16,
        # === Layer layout ===
        hcl_layer_idxs: Optional[List[int]] = None,
        hcm_layer_idxs: Optional[List[int]] = None,
        hcs_layer_idxs: Optional[List[int]] = None,
        attn_layer_idxs: Optional[List[int]] = None,
        # === Convolution / Hyena filter parameters ===
        hcm_filter_length: int = 128,
        hcl_filter_groups: int = 4096,
        hcm_filter_groups: int = 256,
        hcs_filter_groups: int = 256,
        hcs_filter_length: int = 7,
        short_filter_length: int = 3,
        short_filter_bias: bool = False,
        proj_groups: int = 1,
        hyena_filter_groups: int = 1,
        column_split_hyena: bool = False,
        column_split: bool = True,
        interleave: bool = True,
        # === Projection / bias knobs ===
        mha_out_proj_bias: bool = True,
        hyena_out_proj_bias: bool = True,
        qkv_proj_bias: bool = False,
        use_fp8_input_projections: bool = True,
        # === Initialisation ===
        mlp_init_method: str = "torch.nn.init.zeros_",
        mlp_output_init_method: str = "torch.nn.init.zeros_",
        # === Numerical stablility ===
        eps: float = 1e-6,
        # === Rotary ===
        state_size: int = 16,
        rotary_emb_base: int = 100_000_000_000,
        rotary_emb_scaling_factor: int = 128,
        use_interpolated_rotary_pos_emb: bool = True,
        # === Execution-time knobs ===
        max_seqlen: int = 1_048_576,
        max_batch_size: int = 1,
        model_parallel_size: int = 1,
        pipe_parallel_size: int = 1,
        final_norm: bool = True,
        use_flash_attn: bool = True,
        use_flash_rmsnorm: bool = False,
        use_flash_depthwise: bool = False,
        use_flashfft: bool = False,
        use_laughing_hyena: bool = False,
        inference_mode: bool = True,
        tokenizer_type: str = "CharLevelTokenizer",
        prefill_style: str = "fft",
        mlp_activation: str = "gelu",
        print_activations: bool = False,
        log_intermediate_values: bool = False,
        # === Misc. ===
        hyena_flip_x1x2: bool = False,
        **kwargs,
    ) -> None:
        # Provide sane defaults for the (possibly mutable) index lists.
        if hcl_layer_idxs is None:
            hcl_layer_idxs = [2, 6, 9, 13, 16, 20, 23, 27, 30]
        if hcm_layer_idxs is None:
            hcm_layer_idxs = [1, 5, 8, 12, 15, 19, 22, 26, 29]
        if hcs_layer_idxs is None:
            hcs_layer_idxs = [0, 4, 7, 11, 14, 18, 21, 25, 28]
        if attn_layer_idxs is None:
            attn_layer_idxs = [3, 10, 17, 24, 31]

        # === Store attributes ===
        # Use `locals()` instead of manual assignment.  This protects us from
        # typos and makes sure new fields get stored automatically.
        params = locals().copy()
        params.pop("self")  # remove bound instance ref
        kwargs_extra = params.pop("kwargs")

        # Hugging Face base class consumes some kwargs (e.g. *bos_token_id*).
        super().__init__(**kwargs_extra)

        # Persist Evo-2 specific hyper-params in the instance dict.
        self.__dict__.update(params)

    # ---------------------------------------------------------------------
    # Convenience helpers (mirroring Together's StripedHyenaConfig)
    # ---------------------------------------------------------------------

    # Commenting this out because PretrainedConfig's to_dict works for our
    # purposes
    # def to_dict(self):
    #     return {attr: getattr(self, attr) for attr in self.__dict__}

    @classmethod
    def from_original_config(cls, config_path: str, **kwargs):
        """Load a config directly from a json file and convert."""
        with open(config_path, "r", encoding="utf-8") as fp:
            original = json.load(fp)
        return cls(**original, **kwargs)
