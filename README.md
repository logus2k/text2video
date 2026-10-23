# text2video — Wan2.2 on an RTX 4090

Text-to-video with [Wan2.2](https://github.com/Wan-Video/Wan2.2) from quantized GGUF weights stored in `models/`, run
in a Docker container on the desktop's NVIDIA RTX 4090 (24 GB, WSL2). Two models are supported, selected with
`--model`:

| `--model` | Model | What | Native output | Default run |
|---|---|---|---|---|
| `14b` (default) | [Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) | Two 14B experts (high-noise, then low-noise), best quality in the family | 1280x720, 16 fps | 81 frames (5 s), 40 steps |
| `5b` | [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) | One dense 5B model, faster, lower quality | 1280x704, 24 fps | 121 frames (5 s), 50 steps |

The project started on a laptop's Intel Arc iGPU; `generate.py --device xpu` still selects that backend, but the
container, the requirements and the defaults target the 4090 only.

## Models (`models/`, not in git)

Shared by both models, in `models/wan2.2/`:

| File | What | Source |
|---|---|---|
| `umt5-xxl-encoder-Q8_0.gguf` | umT5-XXL text encoder, GGUF Q8_0 (6.0 GB) | [city96/umt5-xxl-encoder-gguf](https://huggingface.co/city96/umt5-xxl-encoder-gguf) |
| `diffusers-config/tokenizer/`, `text_encoder/` | Tokenizer and encoder config | [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) |

Wan2.2-TI2V-5B (`--model 5b`), also in `models/wan2.2/`:

| File | What | Source |
|---|---|---|
| `Wan2.2-TI2V-5B-Q8_0.gguf` | Video transformer, GGUF Q8_0 (5.4 GB) | [unsloth/Wan2.2-TI2V-5B-GGUF](https://huggingface.co/unsloth/Wan2.2-TI2V-5B-GGUF) (mirror of QuantStack) |
| `Wan2.2_VAE.safetensors` | Wan2.2 VAE, fp16 (1.4 GB) | same repo, `VAE/` |
| `diffusers-config/` | Model configs, scheduler (~21 MB) | [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) |

Wan2.2-T2V-A14B (`--model 14b`), in `models/wan2.2-t2v-a14b/`:

| File | What | Source |
|---|---|---|
| `HighNoise/Wan2.2-T2V-A14B-HighNoise-Q8_0.gguf` | High-noise expert, GGUF Q8_0 (14.4 GB) | [QuantStack/Wan2.2-T2V-A14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF) |
| `LowNoise/Wan2.2-T2V-A14B-LowNoise-Q8_0.gguf` | Low-noise expert, GGUF Q8_0 (14.4 GB) | same repo |
| `VAE/Wan2.1_VAE.safetensors` | Wan2.1 VAE (254 MB; the 14B models use the 2.1 VAE) | same repo |
| `diffusers-config/` | Model configs, scheduler | [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) |

Lightning 4-step LoRAs (optional, `--lora`), in `models/loras/`:

| Folder | What | Source |
|---|---|---|
| `Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0/` | high/low-noise LoRA pair, rank 64 (2 x 1.17 GB) | [lightx2v/Wan2.2-Lightning](https://huggingface.co/lightx2v/Wan2.2-Lightning) |
| `Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/` | previous version, for comparison | same repo |

The GGUF repos ship no text encoder; city96's umT5-XXL encoder GGUF is the one used with these GGUF Wan models in
ComfyUI-GGUF (Wan2.1 and 2.2 share the same umT5-XXL encoder). City96 recommends Q5_K_M or larger. Smaller A14B
quantizations (Q6_K 11.2 GB, Q5_K_M 10.1 GB per expert) exist in the same repo if memory gets tight.

Download with the `hf` CLI (`pip install huggingface_hub hf-xet`); ~13 GB for the shared files + 5B, ~29 GB for 14B:

```bash
# shared + 5B
hf download unsloth/Wan2.2-TI2V-5B-GGUF Wan2.2-TI2V-5B-Q8_0.gguf VAE/Wan2.2_VAE.safetensors --local-dir models/wan2.2
mv models/wan2.2/VAE/Wan2.2_VAE.safetensors models/wan2.2/ && rmdir models/wan2.2/VAE
hf download city96/umt5-xxl-encoder-gguf umt5-xxl-encoder-Q8_0.gguf --local-dir models/wan2.2
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  model_index.json scheduler/scheduler_config.json text_encoder/config.json transformer/config.json vae/config.json \
  tokenizer/special_tokens_map.json tokenizer/spiece.model tokenizer/tokenizer_config.json tokenizer/tokenizer.json \
  --local-dir models/wan2.2/diffusers-config
# 14B
hf download QuantStack/Wan2.2-T2V-A14B-GGUF HighNoise/Wan2.2-T2V-A14B-HighNoise-Q8_0.gguf \
  LowNoise/Wan2.2-T2V-A14B-LowNoise-Q8_0.gguf VAE/Wan2.1_VAE.safetensors --local-dir models/wan2.2-t2v-a14b
hf download Wan-AI/Wan2.2-T2V-A14B-Diffusers model_index.json scheduler/scheduler_config.json \
  transformer/config.json transformer_2/config.json vae/config.json --local-dir models/wan2.2-t2v-a14b/diffusers-config
```

Generation itself runs fully offline from `models/` (the image sets `HF_HUB_OFFLINE=1`).

## Run

Requirements: Docker with the NVIDIA Container Toolkit. The image is Ubuntu 26.04 + Python 3.14 + the CUDA 13.0 torch
wheel (which bundles the CUDA runtime, so only the driver is needed on the host).

```bash
docker compose build
docker compose run --rm text2video --prompt "A red fox trotting through fresh snow in a pine forest at sunrise"
docker compose run --rm text2video --model 5b --prompt "..."
docker compose run --rm text2video --frames 1 --prompt "..."      # a single frame: a 1280x720 still image
docker compose run --rm text2video --quant Q6_K --lora Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0 --prompt "..."   # 4-step
```

`models/` is mounted read-only and `outputs/` read-write, so videos land in `outputs/<timestamp>.mp4` on the host.

Options (unset ones take the model's defaults): `--width --height`, `--frames` (4k+1), `--steps`, `--guidance`
(14b: high-noise expert), `--guidance-2` (14b: low-noise expert), `--shift` (scheduler flow shift), `--fps`, `--seed`,
`--negative-prompt`, `--out`, `--device`, `--quant` (which GGUF files to load, default Q8_0), `--sampler` (unipc or
euler), `--lora`. The 14B defaults follow the official config: 40 steps, guidance 4.0 / 3.0, shift 12, expert boundary
0.875.

### Image-to-video

`--image <file>` animates a still instead of starting from text. The 5B model uses its own weights for this; the 14B
model switches to its image-to-video expert pair (`models/wan2.2-i2v-a14b/`, same layout as the T2V ones, plus the I2V
Lightning LoRA `Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1`). The output keeps the image's aspect ratio at the model's
native pixel area unless `--width`/`--height` are given; the first output frame reproduces the input.

```bash
docker compose run --rm text2video --frames 1 --out outputs/seed.mp4 --prompt "..."        # 1. make a still with Wan
docker compose run --rm text2video --image outputs/seed.png --prompt "the fox turns and runs off"   # 2. animate it
```

(`ffmpeg -i outputs/seed.mp4 -frames:v 1 outputs/seed.png` extracts the still.)

### Longer videos: `chain.py`

`chain.py` renders one segment per `--prompt`, each continuation starting from the previous segment's last frame via
image-to-video, and joins them (dropping the duplicated seed frame at each seam, optionally blending `--crossfade N`
frames). Everything else on the command line is passed to `generate.py`:

```bash
docker compose run --rm --entrypoint python text2video chain.py \
  --prompt "A red fox trotting through snow at sunrise" \
  --prompt "The fox stops and looks back at the camera" \
  --prompt "The fox runs off between the pine trees" \
  --lora --quant Q6_K --crossfade 4 --out outputs/fox-story.mp4
```

Segments are kept as `<out>-01.mp4`, `<out>-02.mp4`, ... with their last frames as PNGs, and `<out>.json` records the
prompts; `--start N` re-renders from segment N reusing the earlier ones. Expect some drift in colour and detail across
many hops; a reference still reused as `--image` for every segment, or a subject LoRA, limits it.

### Lightning 4-step LoRA (14B)

[lightx2v/Wan2.2-Lightning](https://huggingface.co/lightx2v/Wan2.2-Lightning) distills the 14B model to 4 steps without
classifier-free guidance, which cuts a render by ~10x. One LoRA per expert, 1.17 GB each, in `models/loras/<name>/`:

```bash
hf download lightx2v/Wan2.2-Lightning Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0/high_noise_model.safetensors \
  Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0/low_noise_model.safetensors --local-dir models/loras
```

`--lora <name>` applies them and switches the defaults to the authors' settings: 4 steps, guidance 1.0, shift 5, Euler
sampler with "simple" spacing (timesteps 1000, 937, 833, 625; the expert swap lands after step 2). The LoRA is applied
by hand inside each quantized linear layer (PEFT's adapter costs ~1.3 GiB of extra activations and trips the WSL2
driver). Use `--quant Q6_K` with it: the LoRA weights add 0.7 GiB and Q8_0 no longer fits at 720p.

Other GPU services (LLM, TTS, STT) normally hold most of the 4090's memory; free their VRAM before generating.

### Without Docker

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python generate.py --prompt "..."
```

## How it works

`generate.py` keeps one model on the GPU at a time, so the rest of the 24 GB goes to activations at full 720p:

1. **Text encoder:** umT5-XXL with its linear layers kept in Q8_0 (~7.5 GB on the GPU), streamed from the memory-mapped
   GGUF file. It encodes the prompt and is then freed.
2. **Transformer:** GGUF Q8_0, dequantized layer by layer during denoising, then freed. For the 14B model the
   high-noise expert runs first; a step callback swaps it for the low-noise expert as soon as the next timestep drops
   below the boundary (0.875 x 1000), so the two 14.4 GB experts are never resident together.
3. **VAE:** bf16, decodes the latents into frames. (fp32 decode of a 720p/121-frame clip took 13 min and 22.8 GiB on
   the 4090; bf16 takes 20 s and 11.9 GiB.)

diffusers can't load the original-layout Wan2.2 VAE file or a T5 GGUF directly, so `generate.py` maps those tensor
names itself. The Wan2.1 VAE used by the 14B model loads through diffusers' single-file converter.

Two memory fixes make 720p video possible on the 14B model, both in `generate.py`:

- **Lean transformer block.** diffusers' `WanTransformerBlock.forward` upcasts the whole hidden state (75,600 tokens x
  5120 at 720p/81 frames) to fp32 for every modulation and residual line and keeps several copies alive, ~100 KB per
  token. `generate.py` replaces it with the same arithmetic done in 8192-token chunks: bit-identical output (measured),
  20% less activation memory, no speed cost.
- **`PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`** (set in `docker-compose.yml` and in `generate.py`). PyTorch's
  default caching allocator fragments to 29 GiB reserved for 19 GiB allocated and the WSL2 driver pages;
  `expandable_segments` fixes the fragmentation but its virtual-memory mapping calls fail on WSL2 above ~20.6 GiB and,
  with a LoRA applied, far below that. The driver's stream-ordered allocator keeps reserved within 0.2 GiB of
  allocated and ran 21.1 GiB at driver level without trouble.

### Measured

| Model and setting | GPU | Time | Peak GPU memory |
|---|---|---|---|
| 5B, 640x352, 41 frames, 30 steps | Arc 140V iGPU (laptop, host) | ~10–12 min | 10.5 GiB |
| 5B, 1280x704, 121 frames, 50 steps, fp32 VAE decode | RTX 4090 (container) | 20.5 min: denoise 7.2 min (8.6 s/step), decode 13.2 min | 22.9 GiB |
| 5B, 1280x704, 121 frames, 50 steps, bf16 VAE decode | RTX 4090 (container) | **7.7 min**: denoise 7.2 min, decode 23 s | 11.9 GiB |
| 5B, 1280x704, 1 frame (still image), 50 steps | RTX 4090 (container) | 21 s | 10.5 GiB |
| 14B Q8_0, 1280x720, 1 frame (still image), 40 steps | RTX 4090 (container) | 1.7 min | 14.9 GiB |
| 14B Q8_0, 832x480, 81 frames, 40 steps | RTX 4090 (container) | 17 min (24.8 s/step) | 18.0 GiB |
| 14B Q8_0, 1280x720, 81 frames, 40 steps | RTX 4090 (container) | does not fit even with the fixes below (~21 GiB) | |
| 14B Q6_K, 1280x720, 81 frames, 40 steps | RTX 4090 (container) | **61 min** (90 s/step), decode 12 s | 17.9 GiB |
| 14B Q6_K + Lightning LoRA, 832x480, 81 frames, 4 steps | RTX 4090 (container) | **90 s** | 14.9 GiB |
| 14B Q6_K + Lightning LoRA, 1280x720, 81 frames, 4 steps | RTX 4090 (container) | **3.8 min** (~46 s/step), decode 13 s | 18.6 GiB |
| 5B image-to-video, 1280x704, 121 frames, 50 steps | RTX 4090 (container) | 7.8 min | 11.9 GiB |
| chain.py, 3 x 41 frames, 5B, 30 steps | RTX 4090 (container) | 4.5 min (90 s per segment) | 11.7 GiB |

Same seed, fp32 vs bf16 decode: PSNR 42 dB between the two videos, no visible difference.

### Memory budget on the 4090 (WSL2)

Under WSL2 the NVIDIA driver does not fail with an out-of-memory error when the 24 GB fill up: it pages GPU memory to
system RAM and each step becomes ~50x slower (24 min instead of ~25 s), with `nvidia-smi` pinned at ~24.1 GB. With
`expandable_segments` the failure is a `CUDA driver error: unknown error` instead (the CUDA virtual-memory mapping
calls it relies on fail through WSL2's /dev/dxg path), which is why the `cudaMallocAsync` backend is used. Driver-level
use is PyTorch's reserved memory plus ~2.2 GiB of context; 21.1 GiB at driver level is verified to run at full speed. Measured for the 14B expert at 1280x720 with the lean block: resident weights (14.4 GiB
Q8_0, 11.3 GiB Q6_K) plus ~0.29 GiB per latent frame (4 video frames), so 81 frames need ~6.5 GiB of activations:
Q6_K fits (17.8 GiB peak), Q8_0 does not. Options when something does not fit: a smaller quantization
(`--quant Q6_K`, `Q5_K_M`; the files come from the same repo), 832x480, or fewer frames. To get a loud error instead of
silent paging without `expandable_segments`, set "CUDA - Sysmem Fallback Policy" to "Prefer No Sysmem Fallback" in the
NVIDIA Control Panel on Windows.
