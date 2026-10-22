"""Wan2.2-TI2V-5B text-to-video on an NVIDIA (CUDA) or Intel Arc (XPU) GPU, using the local weights in models/wan2.2/.

    python generate.py --prompt "A red fox trotting through fresh snow at sunrise"

The GPU backend is picked automatically (CUDA if available, otherwise XPU); force one with --device cuda|xpu.

Weights (all local, see README.md):
    Wan2.2-TI2V-5B-Q8_0.gguf     transformer, GGUF Q8_0 (QuantStack / unsloth mirror)
    Wan2.2_VAE.safetensors       Wan2.2 VAE, original key layout (fp16)
    umt5-xxl-encoder-Q8_0.gguf   umT5-XXL text encoder, GGUF Q8_0 (city96)
    diffusers-config/            configs + tokenizer from Wan-AI/Wan2.2-TI2V-5B-Diffusers

Each model is loaded only for its stage and freed afterwards: text encoder -> transformer -> VAE. This keeps the peak
around 10.5 GiB, which is what lets it fit the Arc iGPU (whose memory under WSL2 comes out of WSL's own RAM, ~15 GB)
and leaves room for other services sharing a discrete GPU. GGUF weights stay quantized on the GPU and are streamed from
the memory-mapped file instead of being copied into RAM first.
"""

import argparse
import gc
import re
import time
import warnings
from datetime import datetime
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan, GGUFQuantizationConfig, UniPCMultistepScheduler, WanPipeline, WanTransformer3DModel
from diffusers.quantizers.gguf.utils import GGUFParameter, _replace_with_gguf_linear, dequantize_gguf_tensor
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor
from gguf import GGMLQuantizationType, GGUFReader
from safetensors.torch import load_file
from transformers import AutoTokenizer, UMT5Config, UMT5EncoderModel

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models" / "wan2.2"
CONFIG = MODELS / "diffusers-config"
TRANSFORMER = MODELS / "Wan2.2-TI2V-5B-Q8_0.gguf"
VAE = MODELS / "Wan2.2_VAE.safetensors"
TEXT_ENCODER = MODELS / "umt5-xxl-encoder-Q8_0.gguf"

BACKENDS = ("cuda", "xpu")
DTYPE = torch.bfloat16

# Default negative prompt from the Wan2.2 model card.
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
    "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

# llama.cpp T5 encoder tensor names -> transformers UMT5EncoderModel names.
UMT5_KEYS = {
    "attn_q": "layer.0.SelfAttention.q",
    "attn_k": "layer.0.SelfAttention.k",
    "attn_v": "layer.0.SelfAttention.v",
    "attn_o": "layer.0.SelfAttention.o",
    "attn_rel_b": "layer.0.SelfAttention.relative_attention_bias",
    "attn_norm": "layer.0.layer_norm",
    "ffn_gate": "layer.1.DenseReluDense.wi_0",
    "ffn_up": "layer.1.DenseReluDense.wi_1",
    "ffn_down": "layer.1.DenseReluDense.wo",
    "ffn_norm": "layer.1.layer_norm",
}

# Wan2.2 VAE residual-block layer indices -> diffusers names.
VAE_RESIDUAL_KEYS = {
    "residual.0.gamma": "norm1.gamma",
    "residual.2.": "conv1.",
    "residual.3.gamma": "norm2.gamma",
    "residual.6.": "conv2.",
    "shortcut.": "conv_shortcut.",
}


def pick_device(choice):
    """Return the torch device to run on: the requested backend, or the first available one in BACKENDS order."""
    available = [b for b in BACKENDS if getattr(torch, b, None) is not None and getattr(torch, b).is_available()]
    if choice == "auto":
        if not available:
            raise SystemExit("No CUDA or XPU device found. Check the GPU runtime (see README.md).")
        return torch.device(available[0])
    if choice not in available:
        raise SystemExit(f"Requested --device {choice} but it is not available (found: {available or 'none'}).")
    return torch.device(choice)


def umt5_key(name):
    if name == "token_embd.weight":
        return "shared.weight"
    if name == "enc.output_norm.weight":
        return "encoder.final_layer_norm.weight"
    block, layer, param = re.fullmatch(r"enc\.blk\.(\d+)\.(\w+)\.(\w+)", name).groups()
    return f"encoder.block.{block}.{UMT5_KEYS[layer]}.{param}"


def gguf_tensors(path, device):
    """Yield (name, tensor) from a GGUF file, moved straight from the memory-mapped file to the GPU."""
    for t in GGUFReader(path).tensors:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # memmap is read-only; it's copied by .to() anyway
            data = torch.from_numpy(t.data).to(device)
        if t.tensor_type in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
            yield t.name, data.to(DTYPE)
        else:
            yield t.name, GGUFParameter(data, quant_type=t.tensor_type)


def load_text_encoder(device):
    """Build UMT5EncoderModel from the city96 GGUF. Linear layers stay Q8_0 and are dequantized on the fly."""
    with torch.device("meta"):
        model = UMT5EncoderModel(UMT5Config.from_pretrained(CONFIG / "text_encoder"))
    state = {umt5_key(name): tensor for name, tensor in gguf_tensors(TEXT_ENCODER, device)}
    # Dequantized up front: nn.Embedding can't use GGUF weights, and UMT5 casts the input of each FFN "wo" layer to
    # wo.weight.dtype, which would be uint8 for a GGUF weight.
    for key in [k for k in state if k == "shared.weight" or k.endswith(".wo.weight")]:
        state[key] = dequantize_gguf_tensor(state[key]).to(DTYPE)
    state["encoder.embed_tokens.weight"] = state["shared.weight"]
    _replace_with_gguf_linear(model, DTYPE, state)
    # Assign directly: load_state_dict() would reject the quantized (byte-shaped) GGUF weights.
    for key, tensor in state.items():
        module, _, param = key.rpartition(".")
        if not isinstance(tensor, torch.nn.Parameter):
            tensor = torch.nn.Parameter(tensor, requires_grad=False)
        model.get_submodule(module)._parameters[param] = tensor
    missing = [n for n, t in [*model.named_parameters(), *model.named_buffers()] if t.is_meta]
    if missing:
        raise RuntimeError(f"Text encoder weights missing from {TEXT_ENCODER.name}: {missing[:5]}")
    return model.eval()


def vae_key(key, num_res_blocks):
    """Map an original Wan2.2 VAE key to the diffusers AutoencoderKLWan layout."""
    key = re.sub(r"^conv1\.", "quant_conv.", key)
    key = re.sub(r"^conv2\.", "post_quant_conv.", key)
    key = re.sub(r"^(encoder|decoder)\.conv1\.", r"\1.conv_in.", key)
    key = key.replace(".head.0.gamma", ".norm_out.gamma").replace(".head.2.", ".conv_out.")
    key = re.sub(r"\.middle\.0\.", ".mid_block.resnets.0.", key)
    key = re.sub(r"\.middle\.1\.", ".mid_block.attentions.0.", key)
    key = re.sub(r"\.middle\.2\.", ".mid_block.resnets.1.", key)

    def block(match, kind, n_resnets, sampler):
        i, j = match.group(1), int(match.group(2))
        return f".{kind}.{i}.resnets.{j}." if j < n_resnets else f".{kind}.{i}.{sampler}."

    key = re.sub(r"\.downsamples\.(\d+)\.downsamples\.(\d+)\.",
                 lambda m: block(m, "down_blocks", num_res_blocks, "downsampler"), key)
    key = re.sub(r"\.upsamples\.(\d+)\.upsamples\.(\d+)\.",
                 lambda m: block(m, "up_blocks", num_res_blocks + 1, "upsampler"), key)
    for old, new in VAE_RESIDUAL_KEYS.items():
        key = key.replace(old, new)
    return key


def load_vae_config():
    config = AutoencoderKLWan.load_config(CONFIG, subfolder="vae")
    config.pop("clip_output", None)
    return config


def load_vae(device):
    # diffusers' single-file converter only knows the Wan2.1 VAE layout, so the Wan2.2 keys are mapped here.
    with torch.device("meta"):
        vae = AutoencoderKLWan.from_config(load_vae_config())
    n = vae.config.num_res_blocks
    state = {vae_key(k, n): v.to(device, torch.float32) for k, v in load_file(VAE).items()}
    vae.load_state_dict(state, assign=True)
    return vae.eval()


def decode(latents, device):
    vae = load_vae(device)
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents.device, torch.float32)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents.device, torch.float32)
    with torch.no_grad():
        video = vae.decode(latents.float() * std + mean, return_dict=False)[0]
    return VideoProcessor(vae_scale_factor=vae.config.scale_factor_spatial).postprocess_video(video, output_type="np")[0]


def load_transformer():
    return WanTransformer3DModel.from_single_file(
        str(TRANSFORMER),
        config=str(CONFIG),
        subfolder="transformer",
        quantization_config=GGUFQuantizationConfig(compute_dtype=DTYPE),
        torch_dtype=DTYPE,
    )


def free_gpu(device):
    gc.collect()
    torch.get_device_module(device).empty_cache()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--prompt", default="A red fox trotting through fresh snow in a pine forest at sunrise, cinematic, soft golden light")
    p.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
    p.add_argument("--device", choices=("auto", *BACKENDS), default="auto",
                   help="GPU backend; auto picks CUDA if available, otherwise XPU (default: auto)")
    # The model is trained at 1280x704 / 704x1280. Smaller sizes (multiples of 32) work and are much faster on an iGPU.
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=352)
    p.add_argument("--frames", type=int, default=41, help="4k+1 frames; 121 = 5 s at 24 fps")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--out", type=Path, help="output .mp4 (default: outputs/<timestamp>.mp4)")
    return p.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)
    backend = torch.get_device_module(device)
    print(f"Device: {device.type} ({backend.get_device_name(0)})")
    out = args.out or ROOT / "outputs" / f"{datetime.now():%Y%m%d-%H%M%S}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("[1/4] Encoding prompt")
    pipe = WanPipeline(
        tokenizer=AutoTokenizer.from_pretrained(CONFIG / "tokenizer"),
        text_encoder=load_text_encoder(device),
        vae=None,  # loaded only for decoding, see decode()
        scheduler=UniPCMultistepScheduler.from_pretrained(CONFIG, subfolder="scheduler"),
        expand_timesteps=True,  # Wan2.2 TI2V
    )
    vae_config = load_vae_config()
    pipe.vae_scale_factor_spatial = vae_config["scale_factor_spatial"]
    pipe.vae_scale_factor_temporal = vae_config["scale_factor_temporal"]
    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            do_classifier_free_guidance=args.guidance > 1.0,
            device=device,
            dtype=DTYPE,
        )
    pipe.register_modules(text_encoder=None)
    free_gpu(device)

    print(f"[2/4] Loading transformer ({time.time() - t0:.0f}s)")
    pipe.register_modules(transformer=load_transformer().to(device))
    free_gpu(device)

    print(f"[3/4] Denoising {args.frames} frames at {args.width}x{args.height}, {args.steps} steps ({time.time() - t0:.0f}s)")
    latents = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        width=args.width,
        height=args.height,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=torch.Generator("cpu").manual_seed(args.seed),
        output_type="latent",
    ).frames
    pipe.register_modules(transformer=None)
    free_gpu(device)

    print(f"[4/4] Decoding ({time.time() - t0:.0f}s)")
    video = decode(latents, device)

    export_to_video(video, str(out), fps=args.fps)
    print(f"Saved {out} ({time.time() - t0:.0f}s total, peak GPU memory {backend.max_memory_allocated() / 2**30:.1f} GiB)")


if __name__ == "__main__":
    main()
