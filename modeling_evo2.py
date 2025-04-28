# New file; must author myself.
# This is an HF PreTrainedModel wrapper.
# Should borrow heavily from https://huggingface.co/togethercomputer/evo-1-131k-base/blob/main/modeling_hyena.py

"""
Hugging Face wrapper for Evo-2 (StripedHyena-2) models.
Drop-in replacement for transformers.AutoModelForCausalLM.
Assumes that the low-level components (Model, layers, etc.) live in the same
flat directory (copied from the Vortex fork) and that custom Triton kernels
were *removed* in favour of Flash-Attention.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_evo2 import Evo2Config  # local import
from .model import StripedHyena                      # Vortex backbone (pure Python)
from .utils import dotdict                           # tiny helper from vortex/utils.py

logger = logging.get_logger(__name__)


class Evo2PreTrainedModel(PreTrainedModel):
    """Base class that defines Hugging Face-specific knobs shared by all Evo-2 wrappers."""

    config_class = Evo2Config
    base_model_prefix = "evo2"
    # Following Together
    supports_gradient_checkpointing = False
    _no_split_modules = ["AttentionBlock", "ParallelGatedConvBlock"]
    _skip_keys_device_placement = ("past_key_values",)
    _keys_to_ignore_on_load_missing = [r"freq"]
    _keys_to_ignore_on_load_unexpected = [r"fftconv", r"twiddle_factors"]
    _supports_flash_attn_2 = True


class Evo2ForCausalLM(Evo2PreTrainedModel):
    """PreTrainedModel-compatible language-model wrapper.

    This is intentionally *thin* - all heavy lifting sits in model.py.
    The wrapper only deals with:
    - HF-style config → dotdict
    - Loss computation
    - Past-key-value cache plumbing so generate() works.
    """

    def __init__(self, config: Evo2Config, **kwargs):
        super().__init__(config, **kwargs)

        # Vortex uses a simple dotdict config object.
        model_cfg = dotdict(config.to_dict())
        self.backbone = StripedHyena(model_cfg)  # ≈ StripedHyena-2 graph
        self.backbone.gradient_checkpointing = False

        # keep a local copy - some layers pad vocab for divisibility
        self.vocab_size = int(config.vocab_size)

        # completes HF initialisation (handles weight tying, etc.)
        self.post_init()

    # Utility helpers expected by transformers.generate
    def _set_gradient_checkpointing(self, enable: bool, _func=None):
        self.backbone.gradient_checkpointing = enable

    def get_input_embeddings(self):
        # backbone hangs onto the token embed as embedding_layer
        return self.backbone.embedding_layer

    # Forward pass (training + generation)
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[dict] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if use_cache:
            # the original Together code has self.backbone.training instead,
            # which is functionally equivalent
            if self.backbone.gradient_checkpointing and self.training:
                logger.warning_once("use_cache=True is incompatible with gradient checkpointing; disabling cache...")
                use_cache = False
            if labels is not None:
                logger.warning_once("use_cache=True is incompatible with loss calculation; disabling cache...")
                use_cache = False

        inputs = input_ids
        if use_cache:
            # Initialise or update specialised inference-state container used by Vortex.
            if past_key_values is None:
                past_key_values = self.backbone.initialize_inference_params()
                batch_size = input_ids.shape[0]
                past_key_values["mha"].max_batch_size = batch_size
                # This line is inherited from Together's HF code. It needs to
                # change for Evo 2 (specifically, we need to access hcl, hcm and
                # hcs instead).
                # past_key_values["hyena"].max_batch_size = batch_size
                past_key_values["hcl"].max_batch_size = batch_size
                past_key_values["hcm"].max_batch_size = batch_size
                past_key_values["hcs"].max_batch_size = batch_size
            else:
                seqlen_offset = past_key_values["mha"].seqlen_offset
                if seqlen_offset == 0:
                    # second loop through generate will have prompt_len + 1 as seqlen
                    seqlen_offset = input_ids.shape[-1] - 1
                    # past_key_values["hyena"].seqlen_offset = seqlen_offset
                    past_key_values["hcl"].seqlen_offset = seqlen_offset
                    past_key_values["hcm"].seqlen_offset = seqlen_offset
                    past_key_values["hcs"].seqlen_offset = seqlen_offset
                    past_key_values["mha"].seqlen_offset = seqlen_offset
                else:
                    past_key_values["mha"].seqlen_offset += 1
                    # past_key_values["hyena"].seqlen_offset += 1
                    past_key_values["hcl"].seqlen_offset += 1
                    past_key_values["hcs"].seqlen_offset += 1
                    past_key_values["hcm"].seqlen_offset += 1

                inputs = input_ids[
                    :,
                    -1:,
                ]

        logits, past_key_values = self.backbone(
            inputs,
            padding_mask=attention_mask,
            inference_params_dict=past_key_values if use_cache else None,
        )

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = F.cross_entropy(shift_logits, shift_labels)

        if return_dict:
            return CausalLMOutputWithPast(
                logits=logits,
                hidden_states=None,
                past_key_values=past_key_values if use_cache else None,
                loss=loss,
            )
        else:
            return logits

    @classmethod
    def can_generate(cls) -> bool:
        return True

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[dict] = None,
        **kwargs,
    ):
        # HF passes past_key_values between generation steps.
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }

    # def _reorder_cache(self, past_key_values: dict, beam_idx: torch.LongTensor) -> dict:
    #     """Reorders cache during beam-search.  Vortex caches pack tensors inside
    #     nested objects; we only need to index along batch dim in-place."""
    #     if past_key_values is None:
    #         return None
    #     for module_cache in past_key_values.values():
    #         for attr, tensor in module_cache.__dict__.items():
    #             if isinstance(tensor, torch.Tensor) and tensor.size(0) == beam_idx.size(0):
    #                 module_cache.__dict__[attr] = tensor.index_select(0, beam_idx)
    #     return past_key_values
