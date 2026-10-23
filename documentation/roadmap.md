# text2video roadmap

State on 2026-10-23: Wan2.2 T2V-A14B and TI2V-5B run from GGUF in one CUDA container on the RTX 4090 (see README for
measured timings). Everything below is ordered by value per effort. Each item lists what it needs and how it is verified.

## 1. Lightning 4-step LoRA (done 2026-10-23)

Result: 480p/81 frames in 90 s (was 17 min), 720p/81 frames in 3.8 min (was 61 min), Q6_K experts, 18.6 GiB peak at
720p. Frames compared against the 40-step renders of the same prompt: same quality class. Needed on the way:
`PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` (expandable_segments fails on WSL2 with the LoRA applied) and a
hand-rolled, allocation-free LoRA instead of PEFT. V1.1 vs V2.0 comparison still open.

Original plan:

Goal: cut the 14B render from 40 steps to 4 (720p clip from 61 min to ~7 min, 480p from 17 min to ~2.5 min).

- Weights: `lightx2v/Wan2.2-Lightning`, folders `Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0` and `-V1.1`, one
  LoRA per expert (`high_noise_model.safetensors`, `low_noise_model.safetensors`, 1.17 GB each), into `models/loras/`.
- Settings from the authors' ComfyUI workflow: 4 steps, CFG 1.0 (no negative pass), shift 5.0, Euler sampler, high
  LoRA for steps 1-2, low LoRA for steps 3-4, strength 1.0. With shift 5 and boundary 0.875 the swap lands after step 2.
- Code: `--lora <dir>` in generate.py; defaults switch to the settings above; LoRA applied to each expert when loaded.
- Verify: (a) LoRA attaches to the GGUF-quantized expert (PEFT on GGUFLinear) and memory stays under 18 GiB;
  (b) 480p clip, then 720p clip, same prompt/seed as the 40-step renders; compare frames side by side.
- Risk: quality loss on large motion (authors' caveat); V2.0 vs V1.1 to be compared on the same seed.

## 2. Image-to-video (done 2026-10-23)

`--image` in generate.py. 5B verified: first frame matches the input (PSNR 38.6 dB), 7.8 min at 1280x704/121 frames.
14B verified with the I2V Q6_K experts and the I2V Lightning LoRA: 4.8 min at 1280x720/81 frames, 19.9 GiB peak,
first frame PSNR 34.2 dB against the input.

Original plan:

Goal: animate a chosen still; prerequisite for clip chaining.

- 5B: no download; the TI2V-5B weights do both. Needs `WanImageToVideoPipeline` wiring with `expand_timesteps` and
  the Wan2.2 VAE encoding the first frame (no CLIP image encoder for this model).
- 14B: `QuantStack/Wan2.2-I2V-A14B-GGUF` HighNoise/LowNoise Q6_K (22 GB) + `Wan-AI/Wan2.2-I2V-A14B-Diffusers`
  configs; Lightning I2V LoRA exists too (`Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1`).
- Code: `--image <file>` selects the I2V pipeline; the image is resized to the target size, aspect preserved.
- Verify: animate `outputs/fox-still-14b.png`; first frame of the output must match the input.

## 3. Longer videos by chaining (step A done 2026-10-23; step B VACE done 2026-10-23)

Step B: `--model vace --extend <clip> --overlap 9` and `chain.py --mode vace`. Wan2.1 VACE 14B Q6_K + Wan2.1 4-step
LoRA at 480p: 112 s per continuation, 19.2 GiB peak. Verified: the extension reproduces the 9 context frames (PSNR
31-33 dB) and the motion continues through the seam (same gait, lighting, background). Needed on the way: lazy VACE
control hints (bit-identical, -2 GiB), VAE tiling + pool release for the conditioning encode, a text-encoder reference
leak fixed in generate.py.

chain.py verified with a 3-segment 5B chain (41 frames each): seams continuous (PSNR 32.8 dB between the seed frame
and the next segment's first frame), scene and subject preserved. Observed limits, as expected from last-frame
chaining: motion restarts at each seam, a later prompt ("turns and runs away") was only partly followed when it
contradicted the incoming motion, and exposure/saturation drift slightly across hops. Step B (VACE overlap
conditioning) remains the answer for invisible seams.

Original plan:

Goal: N prompts -> one continuous MP4.

- Step A (last-frame chaining): a `chain.py` that takes a prompt list, renders clip 1 from text, then each next clip
  from the previous clip's last frame via image-to-video, and joins them with ffmpeg (concat + 4-8 frame cross-fade,
  optional 16->24/30 fps interpolation). Needs item 2.
- Step B (overlap conditioning): Wan2.1 VACE 14B (separate weights) to continue motion from the last 8-16 frames
  instead of one frame; invisible seams. Also unlocks video-to-video editing and reference-image control.
- Consistency across many segments: reuse one reference still as the image input for every segment, or train a
  subject LoRA.
- Verify: 3-segment chain, inspect the two seams frame by frame.

## 4. Prompt extension (done 2026-10-23)

`--extend-prompt` / `--extend-only` in generate.py, `prompt_extend.py` with the official Wan system prompts, calling
agent_server's llama-vision (Gemma 4, multimodal) through LLM_URL. Text and image-aware rewrites verified live.

Original plan:

Goal: better adherence from short prompts, as the official pipeline does with Qwen.

- Use the local LLM (agent_server / llama) to rewrite the prompt into a detailed shot description before generation;
  `--extend-prompt` flag, prints the rewritten prompt.
- Verify: same seed with and without extension, side by side.

## 5. Post-processing helpers

- Frame interpolation (RIFE) 16 -> 32/48 fps; video upscaling (Real-ESRGAN / SeedVR) for 480p previews -> 1080p.
- Both as ffmpeg-friendly scripts or a small container each, following the assets/ service pattern.

## 6. Speech-to-video (optional)

- Wan2.2 S2V-14B: reference image + audio -> talking/gesturing person. Expensive; avatar_server (Ditto) already covers
  presenter shots in real time. Only worth it for full-body or styled shots.

## Open issues

- Q8_0 14B at 720p/81 frames: under cudaMallocAsync the full-size forward runs at full speed (44 s, 21.0 GiB
  allocated, 22.3 reserved, driver pinned at 24.0 GiB), measured twice. A complete 40-step render with it is untested;
  with the LoRA it will not fit (+0.7 GiB). Q6_K stays the 720p default.
- Lightning V1.1 vs V2.0 compared at 480p, same seed: different trajectories (PSNR 12.6 dB), both clean; V2.0 kept
  as default per the authors' ordering.
- `models/wan2.2-t2v-a14b/*/Wan2.2-T2V-A14B-*-Q5_K_M.gguf` (20 GB) were downloaded as a fallback and are unused;
  delete when disk is needed.
- `outputs/fox-720p.mp4` is root-owned (made before the container ran as the host user).
- XPU (laptop) path: generate.py still auto-detects XPU but the container and requirements are CUDA-only; untested
  since the laptop was dropped.
