"""Wan2.2 text-to-video on an NVIDIA GPU (RTX 4090), from local GGUF weights in models/.

    python generate.py --prompt "A red fox trotting through fresh snow at sunrise"             # Wan2.2-T2V-A14B
    python generate.py --model 5b --prompt "..."                                              # Wan2.2-TI2V-5B

Two models (see README.md for the files):
    14b  Wan2.2-T2V-A14B: two 14B "experts" (high-noise, then low-noise), Q8_0 GGUF, Wan2.1 VAE, 1280x720 @ 16 fps.
    5b   Wan2.2-TI2V-5B: one dense 5B model, Q8_0 GGUF, Wan2.2 VAE, 1280x704 @ 24 fps. Faster, lower quality.
Both share the umT5-XXL text encoder (GGUF Q8_0) and tokenizer in models/wan2.2/.

The GPU backend is picked automatically (CUDA if available, otherwise XPU); force one with --device cuda|xpu.

The GPU holds one model at a time: text encoder -> transformer (14b: high-noise expert, swapped for the low-noise one
at the boundary step) -> VAE. GGUF weights stay quantized on the GPU, dequantized layer by layer during denoising, and
are streamed from the memory-mapped file instead of being copied into RAM first.
"""

import argparse
import gc
import os
import re
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Must be set before CUDA initializes. PyTorch's default caching allocator fragments badly on the 14B model (29 GiB
# reserved for 19 GiB allocated at 720p/81 frames) and the WSL2 driver then silently pages; expandable_segments fixes
# the fragmentation but its virtual-memory mapping calls fail on WSL2 ("CUDA driver error: unknown error") above
# ~20.6 GiB and, with a LoRA applied, far below that. The driver's own stream-ordered allocator has neither problem:
# measured 18.6 GiB allocated / 18.8 reserved / 21.1 at driver level at 720p x 81 frames with the Lightning LoRA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

import torch
import diffusers.models.transformers.transformer_wan as transformer_wan
from diffusers import (AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, GGUFQuantizationConfig, UniPCMultistepScheduler,
                       WanImageToVideoPipeline, WanPipeline, WanTransformer3DModel)
from PIL import Image
from diffusers.quantizers.gguf.utils import GGUFParameter, _replace_with_gguf_linear, dequantize_gguf_tensor
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor
from gguf import GGMLQuantizationType, GGUFReader
from safetensors.torch import load_file
from transformers import AutoTokenizer, UMT5Config, UMT5EncoderModel

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
LORAS = MODELS / "loras"
SHARED = MODELS / "wan2.2"  # text encoder + tokenizer, shared by both models
TEXT_ENCODER = SHARED / "umt5-xxl-encoder-Q8_0.gguf"
TOKENIZER = SHARED / "diffusers-config" / "tokenizer"
TEXT_ENCODER_CONFIG = SHARED / "diffusers-config" / "text_encoder"

BACKENDS = ("cuda", "xpu")
DTYPE = torch.bfloat16


@dataclass(frozen=True)
class ModelSpec:
    dir: Path
    transformer: str
    vae: str
    width: int
    height: int
    frames: int
    steps: int
    guidance: float
    fps: int
    shift: float
    transformer_2: str | None = None  # low-noise expert (14b)
    guidance_2: float | None = None   # guidance for the low-noise expert
    boundary: float | None = None     # high->low expert switch, fraction of num_train_timesteps
    expand_timesteps: bool = False    # per-token timesteps (Wan2.2 TI2V)
    wan22_vae: bool = False           # Wan2.2 VAE file (original key layout, mapped by load_vae_22)
    lora: str | None = None           # default Lightning LoRA folder name under models/loras/

    @property
    def config(self):
        return self.dir / "diffusers-config"


# Defaults from the official Wan2.2 configs (wan/configs/wan_t2v_A14B.py: shift 12, 40 steps, guidance 4.0 high / 3.0
# low, boundary 0.875) and the Diffusers model cards (sizes, frames, fps). The 5B scheduler ships with flow_shift 5.0.
SPECS = {
    "14b": ModelSpec(
        dir=MODELS / "wan2.2-t2v-a14b",
        transformer="HighNoise/Wan2.2-T2V-A14B-HighNoise-{quant}.gguf",
        transformer_2="LowNoise/Wan2.2-T2V-A14B-LowNoise-{quant}.gguf",
        vae="VAE/Wan2.1_VAE.safetensors",
        width=1280, height=720, frames=81, steps=40, guidance=4.0, guidance_2=3.0, fps=16, shift=12.0, boundary=0.875,
        lora="Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0",
    ),
    # Image-to-video variant of the 14B model (separate experts, 36 input channels: latents + first-frame condition).
    # boundary 0.9 is from the Diffusers model_index.json.
    "14b-i2v": ModelSpec(
        dir=MODELS / "wan2.2-i2v-a14b",
        transformer="HighNoise/Wan2.2-I2V-A14B-HighNoise-{quant}.gguf",
        transformer_2="LowNoise/Wan2.2-I2V-A14B-LowNoise-{quant}.gguf",
        vae="VAE/Wan2.1_VAE.safetensors",
        width=1280, height=720, frames=81, steps=40, guidance=4.0, guidance_2=3.0, fps=16, shift=12.0, boundary=0.9,
        lora="Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1",
    ),
    "5b": ModelSpec(
        dir=SHARED,
        transformer="Wan2.2-TI2V-5B-{quant}.gguf",
        vae="Wan2.2_VAE.safetensors",
        width=1280, height=704, frames=121, steps=50, guidance=5.0, fps=24, shift=5.0,
        expand_timesteps=True, wan22_vae=True,
    ),
}

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
        model = UMT5EncoderModel(UMT5Config.from_pretrained(TEXT_ENCODER_CONFIG))
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


def vae22_key(key, num_res_blocks):
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


def load_vae_config(spec):
    config = AutoencoderKLWan.load_config(spec.config, subfolder="vae")
    config.pop("clip_output", None)
    return config


def load_vae(spec, device):
    # bf16: at 1280x704 x 121 frames the fp32 decode took 795 s and 22.8 GiB on the RTX 4090; bf16 takes 20 s and 11.9 GiB.
    if not spec.wan22_vae:
        # Wan2.1 VAE: diffusers' single-file converter knows this layout.
        return AutoencoderKLWan.from_single_file(
            str(spec.dir / spec.vae), config=str(spec.config), subfolder="vae", torch_dtype=DTYPE
        ).to(device).eval()
    # Wan2.2 VAE: the converter only knows the Wan2.1 layout, so the keys are mapped here.
    with torch.device("meta"):
        vae = AutoencoderKLWan.from_config(load_vae_config(spec))
    n = vae.config.num_res_blocks
    state = {vae22_key(k, n): v.to(device, DTYPE) for k, v in load_file(spec.dir / spec.vae).items()}
    vae.load_state_dict(state, assign=True)
    return vae.eval()


def decode(spec, latents, device, vae=None):
    vae = vae if vae is not None else load_vae(spec, device)
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents.device, torch.float32)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents.device, torch.float32)
    with torch.no_grad():
        video = vae.decode((latents.float() * std + mean).to(DTYPE), return_dict=False)[0]
    return VideoProcessor(vae_scale_factor=vae.config.scale_factor_spatial).postprocess_video(video, output_type="np")[0]


class SimpleFlowEuler(FlowMatchEulerDiscreteScheduler):
    """Euler sampler with ComfyUI's "simple" spacing: sigmas evenly spaced from 1 down to 1/n, then flow-shifted.
    For 4 steps and shift 5 that is t = 999, 937, 833, 625, the schedule the Lightning LoRAs were tuned on."""

    def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None, mu=None, timesteps=None):
        sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1)[:-1].tolist()
        super().set_timesteps(device=device, sigmas=sigmas)


def make_scheduler(spec, sampler, shift):
    base = UniPCMultistepScheduler.from_pretrained(spec.config, subfolder="scheduler")
    if sampler == "unipc":
        return UniPCMultistepScheduler.from_config(base.config, flow_shift=shift)
    return SimpleFlowEuler(num_train_timesteps=base.config.num_train_timesteps, shift=shift)


_LORA_SCRATCH = {}


def _scratch(key, n, device):
    """A reusable bf16 scratch buffer of at least n elements, so the LoRA path allocates nothing per forward."""
    buf = _LORA_SCRATCH.get(key)
    if buf is None or buf.numel() < n or buf.device != device:
        buf = _LORA_SCRATCH[key] = torch.empty(n, dtype=DTYPE, device=device)
    return buf


def apply_lora(transformer, path, strength=1.0):
    """Apply a Wan LoRA (ComfyUI / lightx2v key layout) to the GGUF-quantized transformer.

    Done by hand rather than with PEFT: PEFT's adapter adds ~1.3 GiB of full-size temporaries at 720p, and on WSL2 the
    expandable-segments allocator fails with "CUDA driver error: unknown error" when the LoRA path allocates many
    temporaries per forward. Here each linear layer's forward gets `out += (x @ A^T) @ B^T * strength`, computed in
    sequence chunks through two preallocated scratch buffers, so a forward pass allocates nothing extra and the only
    extra resident memory is the LoRA weights (registered as buffers, so they follow the module to "meta" when the
    expert is released). diffusers' converter folds alpha/rank into A and B, so no further scaling is needed.
    """
    state = WanPipeline.lora_state_dict(str(path))  # -> transformer.<module>.lora_A.weight / lora_B.weight
    pairs = {}
    for key, tensor in state.items():
        module, kind = key.removeprefix("transformer.").rsplit(".lora_", 1)  # kind: "A.weight" / "B.weight"
        pairs.setdefault(module, {})[kind[0]] = tensor
    device = next(transformer.parameters()).device
    for module_name, ab in pairs.items():
        linear = transformer.get_submodule(module_name)
        linear.register_buffer("lora_A", ab["A"].to(device, DTYPE), persistent=False)  # (rank, in)
        linear.register_buffer("lora_B", ab["B"].to(device, DTYPE), persistent=False)  # (out, rank)
        base_forward = linear.forward

        def forward(x, _lin=linear, _base=base_forward, _s=strength):
            out = _base(x)
            A, B = _lin.lora_A, _lin.lora_B
            if x.shape[0] != 1:  # batch > 1 never happens here (CFG runs cond/uncond separately); keep it simple
                return out.add_((x @ A.T) @ B.T, alpha=_s)
            rank, out_features = B.shape[1], B.shape[0]
            for s in range(0, x.shape[1], LEAN_CHUNK):
                n = min(LEAN_CHUNK, x.shape[1] - s)
                t1 = _scratch("t1", LEAN_CHUNK * rank, x.device)[: n * rank].view(1, n, rank)
                t2 = _scratch("t2", LEAN_CHUNK * out_features, x.device)[: n * out_features].view(1, n, out_features)
                torch.matmul(x[:, s:s + n], A.T, out=t1)
                torch.matmul(t1, B.T, out=t2)
                out[:, s:s + n].add_(t2, alpha=_s)
            return out

        linear.forward = forward
    return len(pairs)


# --- Memory-lean transformer block -----------------------------------------------------------------------------------
# diffusers' WanTransformerBlock.forward upcasts the whole (batch, tokens, 5120) hidden state to fp32 for each
# modulation and residual line and keeps several such copies alive, ~100 KB per token. At 1280x720 x 81 frames
# (75,600 tokens) that is the activation peak and, with the 14B expert resident, more than the 4090 can hold without
# paging. This replacement does the same arithmetic in sequence chunks: bit-identical output (max abs diff 0.0 measured),
# 20% less activation memory, no speed cost.
LEAN_CHUNK = 8192  # tokens per chunk


def _slice(t, s, e):
    """Slice a per-token (B, S, D) tensor along S; pass a per-batch (B, 1, D) modulation tensor through."""
    return t[:, s:e] if t.shape[1] > 1 else t


def _norm_modulate(h, norm, scale=None, shift=None):
    """(norm(h.float()) * (1 + scale) + shift).type_as(h), computed in sequence chunks."""
    out = torch.empty_like(h)
    for s in range(0, h.shape[1], LEAN_CHUNK):
        e = s + LEAN_CHUNK
        x = norm(h[:, s:e].float())
        if scale is not None:
            x = x * (1 + _slice(scale, s, e)) + _slice(shift, s, e)
        out[:, s:e] = x.type_as(h)
    return out


def _residual(h, y, gate):
    """(h.float() + y.float() * gate).type_as(h), computed in sequence chunks."""
    out = torch.empty_like(h)
    for s in range(0, h.shape[1], LEAN_CHUNK):
        e = s + LEAN_CHUNK
        out[:, s:e] = (h[:, s:e].float() + y[:, s:e].float() * _slice(gate, s, e)).type_as(h)
    return out


def lean_block_forward(self, hidden_states, encoder_hidden_states, temb, rotary_emb):
    """Drop-in for WanTransformerBlock.forward (diffusers 0.41.0), same math in chunks."""
    if temb.ndim == 4:  # per-token modulation (Wan2.2 TI2V): temb is (B, S, 6, D)
        mods = [t.squeeze(2) for t in (self.scale_shift_table.unsqueeze(0) + temb.float()).chunk(6, dim=2)]
    else:  # per-batch modulation (Wan2.1 / Wan2.2 14B): temb is (B, 6, D)
        mods = list((self.scale_shift_table + temb.float()).chunk(6, dim=1))
    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = mods

    attn_output = self.attn1(_norm_modulate(hidden_states, self.norm1, scale_msa, shift_msa), None, None, rotary_emb)
    hidden_states = _residual(hidden_states, attn_output, gate_msa)
    del attn_output
    attn_output = self.attn2(_norm_modulate(hidden_states, self.norm2), encoder_hidden_states, None, None)
    hidden_states = hidden_states + attn_output
    del attn_output
    ff_output = self.ffn(_norm_modulate(hidden_states, self.norm3, c_scale_msa, c_shift_msa))
    return _residual(hidden_states, ff_output, c_gate_msa)


transformer_wan.WanTransformerBlock.forward = lean_block_forward
# ---------------------------------------------------------------------------------------------------------------------


def load_transformer(spec, filename, quant):
    return WanTransformer3DModel.from_single_file(
        str(spec.dir / filename.format(quant=quant)),
        config=str(spec.config),
        subfolder="transformer",
        quantization_config=GGUFQuantizationConfig(compute_dtype=DTYPE),
        torch_dtype=DTYPE,
    )


def free_gpu(device):
    gc.collect()
    torch.get_device_module(device).empty_cache()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", choices=("14b", "5b"), default="14b", help="which Wan2.2 model to run (default: 14b)")
    p.add_argument("--image", type=Path, help="image-to-video: animate this image (14b: uses the I2V experts; 5b: same weights). "
                                             "Output size follows the image's aspect ratio at the model's native area unless --width/--height are given")
    p.add_argument("--quant", default="Q8_0", help="GGUF quantization of the transformer file(s) in models/, e.g. Q8_0, Q6_K")
    p.add_argument("--lora", nargs="?", const="default",
                   help="14b only: Lightning 4-step LoRA. Bare --lora picks the model's default folder under models/loras/ "
                        "(see SPECS); or give a folder with high_noise_model.safetensors and low_noise_model.safetensors. "
                        "Switches defaults to 4 steps, no guidance, shift 5, euler")
    p.add_argument("--sampler", choices=("unipc", "euler"), help="unipc (default) or euler with 'simple' spacing (default with --lora)")
    p.add_argument("--prompt", default="A red fox trotting through fresh snow in a pine forest at sunrise, cinematic, soft golden light")
    p.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
    p.add_argument("--device", choices=("auto", *BACKENDS), default="auto",
                   help="GPU backend; auto picks CUDA if available, otherwise XPU (default: auto)")
    # Unset options take the model's defaults (see SPECS): native size, 5 s of video, official steps/guidance/shift.
    p.add_argument("--width", type=int, help="multiple of 16 (14b) or 32 (5b); native 1280x720 (14b) / 1280x704 (5b)")
    p.add_argument("--height", type=int)
    p.add_argument("--frames", type=int, help="4k+1 frames; 1 = a still image. Default 81 (14b) / 121 (5b) = 5 s")
    p.add_argument("--steps", type=int)
    p.add_argument("--guidance", type=float, help="classifier-free guidance (14b: for the high-noise expert)")
    p.add_argument("--guidance-2", type=float, help="14b only: guidance for the low-noise expert")
    p.add_argument("--shift", type=float, help="scheduler flow shift")
    p.add_argument("--fps", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, help="output .mp4 (default: outputs/<timestamp>.mp4)")
    args = p.parse_args()
    if args.image and not args.image.is_file():
        p.error(f"--image {args.image}: no such file")
    spec = SPECS["14b-i2v" if args.image and args.model == "14b" else args.model]
    if args.lora:
        if spec.transformer_2 is None:
            p.error("--lora is only supported with --model 14b")
        if args.lora == "default":
            args.lora = spec.lora
        args.lora = Path(args.lora) if Path(args.lora).is_absolute() else LORAS / args.lora
        for f in ("high_noise_model.safetensors", "low_noise_model.safetensors"):
            if not (args.lora / f).is_file():
                p.error(f"missing {args.lora / f}")
        # Lightning settings (from the authors' ComfyUI workflow): 4 steps, CFG 1.0, shift 5, euler/simple.
        lora_defaults = {"steps": 4, "guidance": 1.0, "guidance_2": 1.0, "shift": 5.0, "sampler": "euler"}
        for name, value in lora_defaults.items():
            if getattr(args, name) is None:
                setattr(args, name, value)
    if args.sampler is None:
        args.sampler = "unipc"
    if args.image and (args.width is None or args.height is None):
        # Keep the image's aspect ratio at the model's native pixel area (as the official CLI does for --size).
        with Image.open(args.image) as im:
            aspect = im.width / im.height
        multiple = 2 * (16 if spec.wan22_vae else 8)  # VAE stride x patch size
        area = spec.width * spec.height
        args.width = args.width or int(round((area * aspect) ** 0.5 / multiple)) * multiple
        args.height = args.height or int(round((area / aspect) ** 0.5 / multiple)) * multiple
    for name in ("width", "height", "frames", "steps", "guidance", "guidance_2", "shift", "fps"):
        if getattr(args, name) is None:
            setattr(args, name, getattr(spec, name))
    return args, spec


def main():
    args, spec = parse_args()
    device = pick_device(args.device)
    backend = torch.get_device_module(device)
    print(f"Device: {device.type} ({backend.get_device_name(0)}), model: {args.model} {args.quant}"
          + (f", LoRA {args.lora.name}" if args.lora else "") + f", {args.sampler}, shift {args.shift}"
          + (f", image-to-video from {args.image}" if args.image else ""))
    out = args.out or ROOT / "outputs" / f"{datetime.now():%Y%m%d-%H%M%S}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("[1/4] Encoding prompt")
    # Image-to-video needs the VAE up front (to encode the first frame), so it stays loaded; text-to-video loads it
    # only for decoding (see decode()).
    vae = load_vae(spec, device) if args.image else None
    pipe = (WanImageToVideoPipeline if args.image else WanPipeline)(
        tokenizer=AutoTokenizer.from_pretrained(TOKENIZER),
        text_encoder=load_text_encoder(device),
        vae=vae,
        scheduler=make_scheduler(spec, args.sampler, args.shift),
        boundary_ratio=spec.boundary,
        expand_timesteps=spec.expand_timesteps,
    )
    # The A14B vae/config.json omits the scale factors (class defaults 8 / 4), so read them off an instantiated config.
    with torch.device("meta"):
        vae_config = AutoencoderKLWan.from_config(load_vae_config(spec)).config
    pipe.vae_scale_factor_spatial = vae_config.scale_factor_spatial
    pipe.vae_scale_factor_temporal = vae_config.scale_factor_temporal
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

    def load_expert(filename, lora_file):
        model = load_transformer(spec, filename, args.quant).to(device)
        if args.lora:
            n = apply_lora(model, args.lora / lora_file)
            print(f"      LoRA {args.lora.name}/{lora_file}: {n} layers")
        return model

    print(f"[2/4] Loading transformer ({time.time() - t0:.0f}s)")
    pipe.register_modules(transformer=load_expert(spec.transformer, "high_noise_model.safetensors"))
    free_gpu(device)

    def swap_expert(pipe, i, t, callback_kwargs):
        """14b: once the next timestep is below the boundary, replace the high-noise expert with the low-noise one."""
        timesteps = pipe.scheduler.timesteps
        boundary = spec.boundary * pipe.scheduler.config.num_train_timesteps
        if pipe.transformer_2 is None and i + 1 < len(timesteps) and timesteps[i + 1] < boundary:
            print(f"\n      switching to the low-noise expert after step {i + 1} ({time.time() - t0:.0f}s)")
            # The pipeline loop still references the high-noise expert (its `current_model` local) while this callback
            # runs, so dropping our reference would not free it. Moving it to the meta device releases the GPU storage.
            pipe.transformer.to("meta")
            pipe.register_modules(transformer=None)
            free_gpu(device)
            pipe.register_modules(transformer_2=load_expert(spec.transformer_2, "low_noise_model.safetensors"))
            free_gpu(device)
        return {}

    print(f"[3/4] Denoising {args.frames} frames at {args.width}x{args.height}, {args.steps} steps ({time.time() - t0:.0f}s)")
    image_kwargs = {"image": Image.open(args.image).convert("RGB")} if args.image else {}
    latents = pipe(
        **image_kwargs,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        width=args.width,
        height=args.height,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        guidance_scale_2=args.guidance_2,
        generator=torch.Generator("cpu").manual_seed(args.seed),
        output_type="latent",
        callback_on_step_end=swap_expert if spec.transformer_2 else None,
    ).frames
    pipe.register_modules(transformer=None, transformer_2=None)
    free_gpu(device)

    print(f"[4/4] Decoding ({time.time() - t0:.0f}s)")
    video = decode(spec, latents, device, vae)

    export_to_video(video, str(out), fps=args.fps)
    print(f"Saved {out} ({time.time() - t0:.0f}s total, peak GPU memory {backend.max_memory_allocated() / 2**30:.1f} GiB)")


if __name__ == "__main__":
    main()
