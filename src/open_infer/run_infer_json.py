import argparse
from contextlib import nullcontext
import gc
import importlib
import json
import logging
import os
import sys
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

VAE_SCALE_FACTOR = 8


def _add_local_src_to_path() -> None:
    # Ensure local repo src/ is importable.
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def clean_memory_on_device(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        torch.xpu.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def normalize_image_size(height: int, width: int) -> Tuple[int, int]:
    # Match original infer_network.py behavior: round down to multiples of 8.
    height = (int(height) // 8) * 8
    width = (int(width) // 8) * 8
    return height, width


def tensor_to_pil_chw(t: torch.Tensor) -> Image.Image:
    # t: C,H,W in [0,1]
    t = t.detach().clamp(0.0, 1.0).to(torch.float32).cpu()
    arr = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal JSON inference for two-frame generation")
    parser.add_argument("--dit", type=str, required=True, help="DiT checkpoint path")
    parser.add_argument("--vae", type=str, required=True, help="VAE checkpoint path")
    parser.add_argument("--text_encoder", type=str, required=True, help="Qwen2.5-VL checkpoint path")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Tokenizer directory or HF model id")
    parser.add_argument("--processor_path", type=str, default=None, help="VL processor directory or HF model id")
    parser.add_argument("--from_json", type=str, required=True, help="Input json path")
    parser.add_argument("--save_path", type=str, required=True, help="Output folder")

    parser.add_argument("--height", type=int, default=864)
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--infer_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--flow_shift", type=float, default=14.5)
    parser.add_argument("--negative_prompt", type=str, default=" ")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--device", type=str, default=None, help="cuda/cpu, default auto")
    parser.add_argument("--text_device", type=str, default=None, help="device for text encoder, default same as --device")
    parser.add_argument("--vae_device", type=str, default=None, help="device for vae encode/decode, default same as --device")

    parser.add_argument("--attn_mode", type=str, default="torch", choices=["flash", "torch", "sageattn", "xformers", "sdpa"])
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--fp8_scaled", action="store_true")
    parser.add_argument("--lora_weight", type=str, nargs="*", default=None, help="LoRA weight path(s)")
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=None, help="LoRA multiplier(s)")
    parser.add_argument("--lora_include_pattern", type=str, default=None, help="Regex include filter for LoRA keys")
    parser.add_argument("--lora_exclude_pattern", type=str, default=None, help="Regex exclude filter for LoRA keys")
    parser.add_argument("--network_module", type=str, default=None, help="LoRA network module, e.g. musubi_tuner.networks.lora_qwen_image")
    parser.add_argument("--network_weights", type=str, default=None, help="LoRA weights for network_module path")
    parser.add_argument("--network_dim", type=int, default=64, help="LoRA rank for dynamic network apply")
    parser.add_argument("--network_alpha", type=float, default=64, help="LoRA alpha for dynamic network apply")
    parser.add_argument("--network_args", type=str, nargs="*", default=None, help="Extra key=value args for network module")
    parser.add_argument("--text_encoder_lora_num", type=int, default=1, help="text encoder lora branch count")
    parser.add_argument("--unet_lora_num", type=int, default=1, help="unet lora branch count")

    parser.add_argument("--resize_control_to_image_size", action="store_true")
    parser.add_argument(
        "--no_resize_control_to_official_size",
        action="store_true",
        help="Disable official control-image bucket resizing (original infer.py uses official resize by default).",
    )
    parser.add_argument("--if_pack", action="store_true", help="Match original infer.py control latent packing behavior")
    return parser.parse_args()


def select_devices(args: argparse.Namespace) -> Tuple[torch.device, torch.device, torch.device]:
    base = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    base_device = torch.device(base)
    text_device = torch.device(args.text_device) if args.text_device else base_device
    vae_device = torch.device(args.vae_device) if args.vae_device else base_device
    return base_device, text_device, vae_device


def build_lora_params(args: argparse.Namespace):
    if not args.lora_weight:
        return None, None

    from safetensors.torch import load_file
    from musubi_tuner.utils.lora_utils import filter_lora_state_dict

    lora_weights_list = []
    for path in args.lora_weight:
        logger.info("Loading LoRA: %s", path)
        lora_sd = load_file(path)
        lora_sd = filter_lora_state_dict(
            lora_sd,
            include_pattern=args.lora_include_pattern,
            exclude_pattern=args.lora_exclude_pattern,
        )
        lora_weights_list.append(lora_sd)

    if args.lora_multiplier is None or len(args.lora_multiplier) == 0:
        lora_multipliers = [1.0] * len(lora_weights_list)
    else:
        lora_multipliers = list(args.lora_multiplier)
        while len(lora_multipliers) < len(lora_weights_list):
            lora_multipliers.append(1.0)
        if len(lora_multipliers) > len(lora_weights_list):
            lora_multipliers = lora_multipliers[: len(lora_weights_list)]

    logger.info("LoRA count=%d, multipliers=%s", len(lora_weights_list), lora_multipliers)
    return lora_weights_list, lora_multipliers


def _module_device_dtype(module) -> Tuple[torch.device, torch.dtype]:
    for p in module.parameters():
        return p.device, p.dtype
    for b in module.buffers():
        return b.device, b.dtype
    return torch.device("cpu"), torch.float32


def _place_dynamic_lora_modules(network, model, text_encoder) -> None:
    text_device, text_dtype = _module_device_dtype(text_encoder)
    unet_device, unet_dtype = _module_device_dtype(model)

    # Keep LoRA modules in their original dtype (typically fp32) to match
    # original infer.py / infer_network.py behavior more closely.
    for lora in getattr(network, "text_encoder_loras", []):
        lora.to(device=text_device)
    for lora in getattr(network, "unet_loras", []):
        lora.to(device=unet_device)

    lora_dtype = None
    if getattr(network, "unet_loras", []):
        lora_dtype = next(network.unet_loras[0].parameters()).dtype
    elif getattr(network, "text_encoder_loras", []):
        lora_dtype = next(network.text_encoder_loras[0].parameters()).dtype

    logger.info(
        "Placed dynamic LoRA modules: text_encoder -> %s/%s, unet -> %s/%s, lora_dtype=%s",
        text_device,
        text_dtype,
        unet_device,
        unet_dtype,
        lora_dtype,
    )


def attach_dynamic_network_lora(args: argparse.Namespace, model, text_encoder, vae):
    if not args.network_weights:
        return None

    if not args.network_module:
        raise ValueError("--network_module is required when --network_weights is set")

    logger.info("Attaching dynamic network LoRA: module=%s weights=%s", args.network_module, args.network_weights)
    try:
        from safetensors.torch import load_file

        sd = load_file(args.network_weights)
        down_keys = sorted([k for k in sd.keys() if ".lora_down." in k and k.endswith(".weight")])
        if down_keys:
            sample_key = down_keys[0]
            rank = int(sd[sample_key].shape[0])
            branch_ids = sorted({k.split(".lora_down.")[1].split(".")[0] for k in down_keys})
            logger.info("LoRA weight inspection: rank=%d, branch_ids=%s, sample_key=%s", rank, branch_ids, sample_key)
    except Exception as ex:
        logger.warning("Failed to inspect LoRA weights: %s", ex)

    network_module = importlib.import_module(args.network_module)

    net_kwargs = {}
    for net_arg in args.network_args or []:
        key, value = net_arg.split("=", 1)
        net_kwargs[key] = value

    if hasattr(network_module, "create_arch_network"):
        network = network_module.create_arch_network(
            1.0,
            args.network_dim,
            args.network_alpha,
            vae,
            [text_encoder],
            model,
            text_encoder_lora_num=args.text_encoder_lora_num,
            unet_lora_num=args.unet_lora_num,
            **net_kwargs,
        )
    else:
        network = network_module.create_network(
            1.0,
            args.network_dim,
            args.network_alpha,
            vae,
            [text_encoder],
            model,
            text_encoder_lora_num=args.text_encoder_lora_num,
            unet_lora_num=args.unet_lora_num,
            **net_kwargs,
        )

    network.apply_to([text_encoder], model, apply_text_encoder=True, apply_unet=True)
    info = network.load_weights(args.network_weights)
    _place_dynamic_lora_modules(network, model, text_encoder)
    logger.info("Loaded network weights: %s", info)
    return network


def load_models(args: argparse.Namespace, base_device: torch.device, text_device: torch.device):
    _add_local_src_to_path()
    from musubi_tuner.qwen_image import qwen_image_model, qwen_image_utils

    lora_weights_list, lora_multipliers = build_lora_params(args)

    dit_dtype = None if args.fp8_scaled else torch.bfloat16
    attn_mode = "torch" if args.attn_mode == "sdpa" else args.attn_mode
    model = qwen_image_model.load_qwen_image_model(
        device=base_device,
        dit_path=args.dit,
        attn_mode=attn_mode,
        split_attn=False,
        loading_device=base_device,
        dit_weight_dtype=dit_dtype,
        fp8_scaled=args.fp8_scaled,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
        num_layers=args.num_layers,
    )
    if not args.fp8_scaled:
        model = model.to(device=base_device, dtype=torch.bfloat16)
    else:
        model = model.to(device=base_device)
    model.eval().requires_grad_(False)

    tokenizer, text_encoder = qwen_image_utils.load_qwen2_5_vl(
        args.text_encoder,
        dtype=torch.bfloat16,
        device=text_device,
        disable_mmap=True,
        tokenizer_path=args.tokenizer_path,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
    )
    text_encoder.eval()
    processor_source = args.processor_path if args.processor_path is not None else args.text_encoder
    vl_processor = qwen_image_utils.load_vl_processor(processor_source)

    vae = qwen_image_utils.load_vae(args.vae, device="cpu", disable_mmap=True)
    vae.eval()

    dynamic_network = attach_dynamic_network_lora(args, model, text_encoder, vae)

    return qwen_image_utils, model, tokenizer, text_encoder, vl_processor, vae, dynamic_network


def prepare_control_inputs(
    qwen_image_utils,
    control_paths: List[str],
    vae,
    vae_device: torch.device,
    resize_to_official: bool,
    resize_to_image_size: Optional[Tuple[int, int]],
    if_pack: bool,
) -> Tuple[List[torch.Tensor], List[np.ndarray]]:
    control_latents_raw: List[torch.Tensor] = []
    control_nps: List[np.ndarray] = []

    if not control_paths:
        return control_latents_raw, control_nps

    vae.to(vae_device)
    for idx, path in enumerate(control_paths):
        img_tensor, img_np, _ = qwen_image_utils.preprocess_control_image(
            path,
            resize_to_official,
            resize_to_image_size,
        )
        with torch.no_grad():
            latent = vae.encode_pixels_to_latents(img_tensor.to(vae_device, vae.dtype))
        latent = latent.to(torch.bfloat16).cpu()

        control_latents_raw.append(latent)
        control_nps.append(img_np)

    vae.to("cpu")
    clean_memory_on_device(vae_device)
    # Match original infer.py behavior exactly:
    # - if_pack=False: keep all control latents as-is
    # - if_pack=True: apply pyramid-like resize with scale=2**(i-1), and resize only when i>1
    if not if_pack:
        return control_latents_raw, control_nps

    packed_controls: List[torch.Tensor] = []
    packed_nps: List[np.ndarray] = []
    for i, latent in enumerate(control_latents_raw):
        b, c, f, h, w = latent.shape
        scale = 2 ** (i - 1)
        new_h = h // scale
        new_w = w // scale
        new_h = max(2, new_h - (new_h % 2))
        new_w = max(2, new_w - (new_w % 2))
        if new_h < 1 or new_w < 1:
            logger.warning("Discard control[%d] due to too small packed latent size: %dx%d", i, new_h, new_w)
            continue
        if i > 1:
            latent_4d = latent.squeeze(2)
            latent_4d = F.interpolate(latent_4d, size=(new_h, new_w), mode="bilinear", align_corners=False)
            latent = latent_4d.unsqueeze(2)
        packed_controls.append(latent)
        packed_nps.append(control_nps[i])

    return packed_controls, packed_nps


def encode_prompt_pair(
    qwen_image_utils,
    tokenizer,
    text_encoder,
    vl_processor,
    prompt: str,
    negative_prompt: str,
    control_nps: List[np.ndarray],
) -> Tuple[torch.Tensor, torch.Tensor]:
    te_device, _ = _module_device_dtype(text_encoder)
    use_autocast = te_device.type in ("cuda", "xpu")
    autocast_ctx = torch.autocast(device_type=te_device.type, dtype=torch.bfloat16) if use_autocast else nullcontext()
    with torch.no_grad(), autocast_ctx:
        if control_nps:
            # Keep behavior aligned with original infer.py: only the first control image
            # is used for prompt embedding in inference path.
            control_for_text = control_nps[0]
            embed, mask = qwen_image_utils.get_qwen_prompt_embeds_with_image_infer(
                vl_processor, text_encoder, prompt, control_for_text
            )
            neg_embed, neg_mask = qwen_image_utils.get_qwen_prompt_embeds_with_image_infer(
                vl_processor, text_encoder, negative_prompt, control_for_text
            )
        else:
            embed, mask = qwen_image_utils.get_qwen_prompt_embeds(tokenizer, text_encoder, prompt)
            neg_embed, neg_mask = qwen_image_utils.get_qwen_prompt_embeds(tokenizer, text_encoder, negative_prompt)

    txt_len = int(mask.to(dtype=torch.bool).sum().item())
    neg_txt_len = int(neg_mask.to(dtype=torch.bool).sum().item())
    embed = embed[:, :txt_len]
    neg_embed = neg_embed[:, :neg_txt_len]
    return embed, neg_embed


def generate_two_frames(
    qwen_image_utils,
    model,
    vae,
    base_device: torch.device,
    vae_device: torch.device,
    embed: torch.Tensor,
    neg_embed: torch.Tensor,
    control_latents: List[torch.Tensor],
    seed: int,
    infer_steps: int,
    height: int,
    width: int,
    guidance_scale: float,
    flow_shift: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    height, width = normalize_image_size(height, width)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    embed = embed.to(base_device, dtype=torch.bfloat16)
    neg_embed = neg_embed.to(base_device, dtype=torch.bfloat16)
    txt_seq_lens = [embed.shape[1]]
    neg_txt_seq_lens = [neg_embed.shape[1]]

    num_channels_latents = model.in_channels // 4
    latents_f = qwen_image_utils.prepare_latents(1, num_channels_latents, height, width, torch.bfloat16, base_device, generator)
    latents_l = qwen_image_utils.prepare_latents(1, num_channels_latents, height, width, torch.bfloat16, base_device, generator)
    split_seq_len = latents_f.shape[1]
    latents = torch.cat([latents_f, latents_l], dim=1)

    r_height = 2 * (height // (VAE_SCALE_FACTOR * 2))
    r_width = 2 * (width // (VAE_SCALE_FACTOR * 2))
    img_shapes = [(1, r_height, r_width // 2)]

    if control_latents:
        img_shapes = [img_shapes + [(1, cl.shape[-2] // 2, cl.shape[-1] // 2) for cl in control_latents]]
        control_latent = [qwen_image_utils.pack_latents(cl) for cl in control_latents]
        control_latent = torch.cat(control_latent, dim=1).to(base_device, dtype=torch.bfloat16)
    else:
        img_shapes = [img_shapes]
        control_latent = None

    sigmas = np.linspace(1.0, 1.0 / infer_steps, infer_steps)
    image_seq_len = latents.shape[1]
    mu = qwen_image_utils.calculate_shift_qwen_image(image_seq_len)
    scheduler = qwen_image_utils.get_scheduler(flow_shift)
    timesteps, n = qwen_image_utils.retrieve_timesteps(scheduler, infer_steps, base_device, sigmas=sigmas, mu=mu)
    assert n == infer_steps

    do_cfg = guidance_scale > 1.0
    scheduler.set_begin_index(0)
    use_autocast = base_device.type in ("cuda", "xpu")

    with tqdm(total=infer_steps, desc="denoising", leave=False) as pbar:
        for i, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            latent_model_input = latents if control_latent is None else torch.cat([latents, control_latent], dim=1)

            autocast_ctx = torch.autocast(device_type=base_device.type, dtype=torch.bfloat16) if use_autocast else nullcontext()
            with torch.no_grad(), autocast_ctx:
                noise_pred = model(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=None,
                    encoder_hidden_states_mask=None,
                    encoder_hidden_states=embed,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                )
                if control_latent is not None:
                    noise_pred = noise_pred[:, :image_seq_len]

            if do_cfg:
                autocast_ctx = torch.autocast(device_type=base_device.type, dtype=torch.bfloat16) if use_autocast else nullcontext()
                with torch.no_grad(), autocast_ctx:
                    neg_noise_pred = model(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states_mask=None,
                        encoder_hidden_states=neg_embed,
                        img_shapes=img_shapes,
                        txt_seq_lens=neg_txt_seq_lens,
                    )
                    if control_latent is not None:
                        neg_noise_pred = neg_noise_pred[:, :image_seq_len]

                comb_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)
                cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                noise_pred = comb_pred * (cond_norm / noise_norm)

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if i == len(timesteps) - 1 or (i + 1) % scheduler.order == 0:
                pbar.update()

    latents_f = latents[:, :split_seq_len]
    latents_l = latents[:, -split_seq_len:]
    latents_f = qwen_image_utils.unpack_latents(latents_f, height, width)
    latents_l = qwen_image_utils.unpack_latents(latents_l, height, width)

    vae.to(vae_device)
    with torch.no_grad():
        pix_f = vae.decode_to_pixels(latents_f.to(vae_device))
        pix_l = vae.decode_to_pixels(latents_l.to(vae_device))
    vae.to("cpu")
    clean_memory_on_device(vae_device)

    return pix_f[0].cpu().to(torch.float32), pix_l[0].cpu().to(torch.float32)


def process_json(args: argparse.Namespace) -> None:
    base_device, text_device, vae_device = select_devices(args)
    os.makedirs(args.save_path, exist_ok=True)

    with open(args.from_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError("Input JSON must be a list.")

    qwen_image_utils, model, tokenizer, text_encoder, vl_processor, vae, _dynamic_network = load_models(
        args, base_device, text_device
    )

    base_seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    logger.info(
        "Loaded %d records. device=%s, text_device=%s, vae_device=%s, base_seed=%d",
        len(records),
        base_device,
        text_device,
        vae_device,
        base_seed,
    )
    logger.info(
        "Inference defaults: height=%d width=%d infer_steps=%d cfg(guidance)_scale=%.3f flow_shift=%.3f if_pack=%s resize_control_to_official=%s",
        args.height,
        args.width,
        args.infer_steps,
        args.guidance_scale,
        args.flow_shift,
        args.if_pack,
        not args.no_resize_control_to_official_size,
    )

    for i, rec in enumerate(records):
        idx = rec.get("idx", i)
        prompt = rec.get("prompt", rec.get("caption", ""))
        if not prompt:
            logger.warning("Skip record %s because prompt/caption is empty.", idx)
            continue

        control_paths = rec.get("control_img_list", []) or []
        negative_prompt = rec.get("negative_prompt", args.negative_prompt)
        seed = rec.get("seed", base_seed + i)
        sample_steps = int(rec.get("sample_steps", args.infer_steps))
        sample_height = int(rec.get("height", args.height))
        sample_width = int(rec.get("width", args.width))
        sample_guidance_scale = float(rec.get("cfg_scale", rec.get("guidance_scale", args.guidance_scale)))
        sample_flow_shift = rec.get("discrete_flow_shift", rec.get("flow_shift", args.flow_shift))

        resize_to_size = (sample_width, sample_height) if args.resize_control_to_image_size else None
        resize_to_official_size = not args.no_resize_control_to_official_size
        control_latents, control_nps = prepare_control_inputs(
            qwen_image_utils,
            control_paths,
            vae,
            vae_device,
            resize_to_official_size,
            resize_to_size,
            args.if_pack,
        )

        with torch.no_grad():
            embed, neg_embed = encode_prompt_pair(
                qwen_image_utils,
                tokenizer,
                text_encoder,
                vl_processor,
                prompt,
                negative_prompt,
                control_nps,
            )

            pix_f, pix_l = generate_two_frames(
                qwen_image_utils,
                model,
                vae,
                base_device,
                vae_device,
                embed,
                neg_embed,
                control_latents,
                seed,
                sample_steps,
                sample_height,
                sample_width,
                sample_guidance_scale,
                sample_flow_shift,
            )

        out_f = os.path.join(args.save_path, f"{idx}_f.png")
        out_l = os.path.join(args.save_path, f"{idx}_l.png")
        tensor_to_pil_chw(pix_f).save(out_f)
        tensor_to_pil_chw(pix_l).save(out_l)
        logger.info("Saved idx=%s -> %s, %s", idx, out_f, out_l)

        clean_memory_on_device(base_device)
        clean_memory_on_device(text_device)


def main() -> None:
    args = parse_args()
    process_json(args)


if __name__ == "__main__":
    main()
