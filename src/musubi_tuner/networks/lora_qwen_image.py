# LoRA module for Qwen-Image

import ast
from typing import Dict, List, Optional
import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import musubi_tuner.networks.lora as lora

# TODO 它只动了QwenImageTransformerBlock，给它加一个textencoder的layer，再改一下lora.py
# 给它加进去
QWEN_IMAGE_TARGET_REPLACE_MODULES = [
    "QwenImageTransformerBlock",
    "Qwen2_5_VLDecoderLayer",  
    # "Qwen2_5_VLVisionBlock"
]
# QWEN_IMAGE_TARGET_REPLACE_MODULES = ["QwenImageTransformerBlock"]

def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    text_encoder_lora_num: int = 1,  # XXX 新增参数，控制text encoder的lora数量
    unet_lora_num: int =2,
    **kwargs,
):
    # add default exclude patterns
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)

    # exclude if '_mod_' in the name of the module (modulation)
    exclude_patterns.append(r".*(_mod_).*")

    kwargs["exclude_patterns"] = exclude_patterns

    return lora.create_network(
        QWEN_IMAGE_TARGET_REPLACE_MODULES,
        "lora_unet",
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        text_encoder_text_encoder_lora_num = text_encoder_lora_num,  # XXX 新增参数，控制text encoder的lora数量
        unet_lora_num=unet_lora_num,
        **kwargs,
    )

# 这块没顺着改，只把输入放进去了(好像我后面给改完了，回头检查一下，真改完了就给这里删掉)
# XXX 新增参数，控制text encoder的lora数量
def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    text_encoder_lora_num: int = 1,
    unet_lora_num: int=2,
    **kwargs,
) -> lora.LoRANetwork:
    return lora.create_network_from_weights(
        QWEN_IMAGE_TARGET_REPLACE_MODULES, multiplier, weights_sd, text_encoders, unet, for_inference, text_encoder_lora_num=text_encoder_lora_num,unet_lora_num=unet_lora_num,**kwargs
    )


