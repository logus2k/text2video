# text2video — Wan2.2-TI2V-5B on Intel Arc (XPU)

Text-to-video with [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) on the laptop's Intel Arc 140V iGPU
(16 GB shared memory), under WSL2, using quantized GGUF weights stored in `models/`.

## Models (`models/wan2.2/`, not in git)

| File | What | Source |
|---|---|---|
| `Wan2.2-TI2V-5B-Q8_0.gguf` | Video transformer, GGUF Q8_0 (5.4 GB) | [unsloth/Wan2.2-TI2V-5B-GGUF](https://huggingface.co/unsloth/Wan2.2-TI2V-5B-GGUF) (mirror of QuantStack) |
| `Wan2.2_VAE.safetensors` | Wan2.2 VAE, fp16 (1.4 GB) | same repo, `VAE/` |
| `umt5-xxl-encoder-Q8_0.gguf` | umT5-XXL text encoder, GGUF Q8_0 (6.0 GB) | [city96/umt5-xxl-encoder-gguf](https://huggingface.co/city96/umt5-xxl-encoder-gguf) |
| `diffusers-config/` | Model configs, scheduler, tokenizer (~21 MB) | [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) |

The GGUF repos ship no text encoder; city96's umT5-XXL encoder GGUF is the one used with these GGUF Wan models in
ComfyUI-GGUF (Wan2.1 and 2.2 share the same umT5-XXL encoder). City96 recommends Q5_K_M or larger.

To re-download everything except the two files that were already there:

```bash
hf download city96/umt5-xxl-encoder-gguf umt5-xxl-encoder-Q8_0.gguf --local-dir models/wan2.2
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --include "model_index.json" "scheduler/*" "tokenizer/*" "text_encoder/config.json" "transformer/config.json" "vae/config.json" \
  --local-dir models/wan2.2/diffusers-config
```

## Setup

1. Intel GPU runtime (Level Zero) inside WSL. Use `--no-install-recommends`: the recommended `*-legacy` driver
   (for pre-Xe2 GPUs) gets loaded too and makes `torch.xpu` segfault on the Arc 140V.

   ```bash
   sudo apt-get install -y --no-install-recommends libze1 libze-intel-gpu1 intel-opencl-icd
   ```

2. Python packages (venv, Python 3.14):

   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

3. Check the GPU is visible (the "Can't initialize Level Zero Sysman" warning is harmless on WSL):

   ```bash
   .venv/bin/python -c "import torch; print(torch.xpu.is_available(), torch.xpu.get_device_name(0))"
   ```

### Corporate proxy

Traffic goes through the corporate proxy (`HTTP(S)_PROXY`, or the local cntlm at `127.0.0.1:3128` used by apt and
`~/.local/bin/net-proxy.sh`). Both routes reach Hugging Face. Python's HTTP libraries use their own CA bundle
(certifi), so point them at the system store, which includes the corporate root CA:

```bash
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```

`huggingface.co` and the Xet download hosts (`*.xethub.hf.co`) are not TLS-inspected. `cdn-lfs.huggingface.co` is
inspected by `swg-lx-01` with a certificate the system store doesn't trust; current downloads use Xet so this hasn't
mattered, but if a download fails with a certificate error, that host is the likely cause.

Generation itself runs fully offline from `models/`.

## Run

```bash
.venv/bin/python generate.py --prompt "A red fox trotting through fresh snow in a pine forest at sunrise"
```

Output goes to `outputs/<timestamp>.mp4`. Options: `--width --height` (multiples of 32), `--frames` (4k+1),
`--steps`, `--guidance`, `--seed`, `--fps`, `--negative-prompt`, `--out`.

Defaults are 640x352, 41 frames (~1.7 s of video), 30 steps. The model is trained at **1280x704** (or 704x1280) with
121 frames (5 s at 24 fps), which gives the best quality but is far slower on this iGPU.

Measured with the defaults on the Arc 140V:

| Stage | Time |
|---|---|
| Startup + encode prompt | ~70 s |
| Load transformer | ~30–160 s (slower when the file isn't in the page cache) |
| Denoise, 30 steps | ~4.5 min (~9 s/step) |
| Decode | ~110 s |
| **Total** | **~10–12 min**, peak 10.5 GiB GPU memory |

### How it fits in memory

Under WSL2, memory allocated on the Arc iGPU comes out of **WSL's own RAM** (15 GB by default, half of the
laptop's 31.5 GB), so CPU and GPU share about 13.5 GB. `generate.py` loads one model per stage and frees it before
the next one:

1. **Text encoder:** umT5-XXL with its linear layers kept in Q8_0 (~7.5 GB on the GPU), streamed from the memory-mapped
   GGUF file. It encodes the prompt and is then freed.
2. **Transformer:** GGUF Q8_0 (~5.2 GB), dequantized layer by layer during denoising, then freed.
3. **VAE:** fp32 (~2.8 GB), decodes the latents into frames.

diffusers can't load the original-layout Wan2.2 VAE file or a T5 GGUF directly, so `generate.py` maps those
tensor names itself.

To give WSL more room (useful for 1280x704), set this in `%USERPROFILE%\.wslconfig` on Windows, then run
`wsl --shutdown`:

```ini
[wsl2]
memory=24GB
```
