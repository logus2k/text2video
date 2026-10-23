# Movie: training videos from Cortex knowledge — analysis and plan

Status: DRAFT for decision, 2026-10-23. Written after reading Cortex (HEAD eb9acd6: README, technical architecture,
the Recording, Decks, Learning, Assistants, v1/OAuth/A2A/MCP code, the GPU release mechanisms, the Red Accent design
system) and after the text2video lab's verified results (see ../README.md). Nothing here is built. The questions a
decision needs are in §1; the plan follows them.

---

## 0. Summary

- **The request makes sense and fits Cortex's own model.** Cortex already produces three kinds of content from the
  domains, each with the same shape of pipeline: gather cited passages → plan with the model → the person reviews →
  build → deliver. Decks (`cortex/decks/`) is the closest analogue of a movie: outline, writer, artist, critic, a
  review step before build, a per-project store with versions, proposals the person accepts. Trainings already turn
  into decks through a hook. A movie is the next output of the same material: a script (narration + visuals) made
  from cited passages, rendered to video.
- **Recommendation: build the Movie as a Cortex channel from the first release, with the video generation in a
  separate GPU service.** Not a separate product frontend. Reasons in §2. The precedent is Decks itself: it started
  as a separate Slides app and had to be merged later (2026-10-11); a Movie app would repeat that merge.
- **GPU capacity is the hard constraint and it is solvable with mechanisms Cortex already has.** The 4090 holds
  about 24 GB; a 14B render needs 15–20 GB; the resident services hold 13–17 GB. The movie job therefore runs inside a
  *GPU lease*: Cortex asks its own services to release the card (the speech lease and the idle-release routes exist;
  one route is missing for the local model), renders, then lets them reload. The engine itself holds nothing when idle.
  §4 has the budget and the sequence. (The cross-app GPU gate rejected on 2026-10-10 is not proposed; this is
  Cortex orchestrating its own tenants, the pattern it already uses for ingestion.)
- **What a person gets:** say what the video is for, pick domains, agree a script in a conversation with the Movie
  Assistant, see the movie as a track of shots (narration + picture per shot), reorder/retake/insert/delete shots,
  render, watch, publish. Realistic cost on this card: a 5-minute training video with ~20 generated shots renders in
  30–60 minutes; stills in seconds; slides and screen recordings cost no GPU.

---

## 1. Decisions needed before building

| # | Question | Options | Recommendation |
|---|---|---|---|
| D1 | Where does the Movie UI live? | (a) Cortex channel (tab + Assistant), (b) separate web app on Red Accent with its own backend calling Cortex's v1 as the person | **(a)**, §2 |
| D2 | Where does rendering run? | (a) inside the Cortex process, (b) a separate service `movie_engine` (this lab, containerised, its own repo under assets/) | **(b)**: different torch (2.14 cu130 vs Cortex's 2.11 cu128), 100+ GB of models, must be 0 GB at idle, same pattern as stt/tts/ingestion |
| D3 | First-release shot kinds | narration + generated clip; still (Ken Burns); slide (from Decks renders); uploaded screen recording / image; presenter (avatar) | all but presenter in release 1; presenter in release 2 (§6) |
| D4 | Narration voice | tts_server (Kokoro; pt = Brazilian voice `pf_dora`, no pt-PT) | use it; note the pt-PT gap as a known issue |
| D5 | Publishing | movie stays in the project; and/or its script is submitted to a domain through the review lifecycle so the video is searchable and cited | both, review optional per movie |
| D6 | AI-generated marking | visible end card + metadata, per the AI Act note already in Cortex's architecture | required in release 1 |
| D7 | Quality tiers | "preview" (4-step, 480p, ~90 s/shot) and "final" (4-step 720p ~4 min/shot; 40-step on request) | preview by default while editing, final at export |
| D8 | Licences | Wan2.2 Apache-2.0; Lightning LoRAs; RIFE (MIT); Real-ESRGAN (BSD-3) / SeedVR2 — to be added to THIRD_PARTY_LICENSES | verify the LoRA licence before release |

---

## 2. Separate frontend or Cortex channel

What a separate app would have to rebuild or bypass, all of which Cortex provides today:

| Need | In Cortex | Separate app |
|---|---|---|
| Sign-in and identity | Entra via oauth2-proxy, `X-Auth-Request-Email` + proxy secret | its own proxy route and oauth2-proxy (pattern exists) |
| Cited content as the person, with access control | in-process `v1` handlers / `kbase.retrieval.search` (used by Decks via `decks/kb.py KnowledgeBase`) | `/api/v1` with an OAuth 2.1 person token (consent page), or a service key + `X-On-Behalf-Of` from the proxy's internal server (`internal-cortex.conf`, port 8710). Both work; no CORS, so the app needs its own backend |
| Projects, notifications, audit, retention, project deletion | `api/projects.py`, `notifications.notify`, `audit.LABELS`, `retention` | none of it over v1 (v1 is read-only, no projects) |
| Assistant pane, chat component, speech pad, notes | `rightPanel.add`, `ChatPanel`, `speech.js`, notes runner | rebuild |
| Design system shell, i18n (EN/PT, ~2,700 strings), themes, screens, phone page | done | rebuild |
| Decks slides as shots, trainings as chapters, review lifecycle for publishing | hooks exist (`decks/hooks.py`, learning → deck) | not reachable over v1 |
| TTS, speech-away degradation, Model Health | `speech_status`, `speech.js` | rebuild |
| Observability standard, tests (Playwright suites, live checks) | done | rebuild |

The separate app's only advantages: freedom in the UI stack (Cortex is plain JS without a bundler) and release
decoupling. The track editor the Movie needs is similar in complexity to the Decks editor, which was built in that
stack, so the stack is not a blocker. Decoupling is obtained instead by putting the heavy part (rendering) in its own
service (D2) with a small, stable HTTP contract (§3.2); the Cortex side is orchestration and UI, like Decks.

If a standalone UI is still wanted for demos or for use without Cortex, build it later as a thin page over the same
engine contract; do not start there.

---

## 3. Architecture

```
 browser ── Cortex page: Movie tab (track editor) + Movie Assistant (right pane)
    │ socket.io / HTTP (same as today)
    ▼
 cortex :2710 ──── Movie channel (cortex/movies/): store, gather+script, jobs, GPU lease, assembly, publish
    ├─▶ kbase.retrieval / v1 (cited passages, as the person)      ├─▶ agent_server (presets cortex_movie_*)
    ├─▶ tts_server (narration audio)                                ├─▶ decks (slide renders), learning (trainings)
    ├─▶ stt/tts/embeddings/ingestion admin routes (release/load)   └─▶ reviews (publish the script into a domain)
    └─▶ movie_engine :2720 (new, cortex-kb network) ── Wan2.2 renders, interpolation, upscaling, ffmpeg assembly
```

### 3.1 The data model

`CORTEX_MOVIES_DIR` (`./movies`), one folder per movie, validated JSON like decks:

```
movies/projects/<pid>/movies/<mid>/
  movie.json         {id, project, title, brief{purpose, audience, language, duration_target, domains[], tone, style},
                      script_version, status, created_by, created_at, ai_marking: true}
  script.json        {chapters[{title, shots[<shot id>]}], shots{<id>: Shot}, sources[{document, link}]}
  shots/<sid>/       narration.wav, prompt.txt, seed.png, clip.mp4, clip-final.mp4, thumb.jpg, status.json
  versions/<n>.mp4   assembled movies (append-only, like deck versions) + <n>.srt + <n>.json (what went in)
  jobs/<jid>.json    {kind, status, progress, shots[], started, finished, error}   (persisted; resumable)
  conversation/      the Movie Assistant thread (messages.jsonl), like decks/conversations
  audit.jsonl
```

A **Shot** is the unit the person edits:

```
{id, kind: clip|still|slide|upload|presenter, narration: "text the voice says", duration: s (from the narration audio,
 min 3 s), visual: {prompt, extended_prompt, seed, image?, deck?, slide?, file?, quality: preview|final},
 sources: [cite ids], transitions: {in, out}, status: {narration: ok|pending, visual: pending|rendering|ok|failed}}
```

Duration follows the narration: the picture must cover the spoken text. A generated clip is 5 s; longer narration
over one shot is covered by slow-motion/interpolation, Ken Burns on the last frame, or by splitting the shot (the
Assistant proposes the split). This is how the 5-second limit of the models becomes invisible to the person.

### 3.2 The engine contract (`movie_engine`, this lab, moved to `assets/movie_engine/`)

HTTP, JSON, bearer token (like ingestion-server), on `cortex-kb`:

| Route | Does |
|---|---|
| `POST /v1/jobs` `{kind: still|clip|i2v|interpolate|upscale|assemble, params…}` → `{job}` | queue one job; the engine runs one GPU job at a time |
| `GET /v1/jobs/{job}` → `{status, progress, result{file,…}, error}` | poll; progress is step-based (the pipeline's step callback) |
| `DELETE /v1/jobs/{job}` | cancel (stops the render container's process) |
| `GET /v1/files/{…}` | the result files (also reachable through a shared volume) |
| `GET /admin/state` `{idle, gpu_mib, loaded}`; `POST /admin/release` | state and an explicit "unload everything now" |
| `GET /health` | |

Rules: **0 GB VRAM at idle** (each job is a subprocess that loads one model per stage and exits; proven by the lab's
per-stage loading and by `nvidia-smi` after a run), one job at a time, model files on a host volume, quality tiers
map to `--quant/--lora/--width/--height/--steps`, every job writes its own `job.json` so a restart loses nothing but
the running job. Contract tests run the fox prompts and check the files exist, have the expected frames and fps, and
that VRAM returns to idle.

### 3.3 From intention to movie (the pipeline, mirrored on Decks)

1. **Intention.** The person opens Movie, says what the video is for (free text or a short form: purpose, audience,
   language, length, domains, tone). The Movie Assistant (preset `cortex_movie_brief`) asks the two or three questions
   that matter and writes the brief.
2. **Gather.** As Decks' `outline.gather`: cited passages for the brief's topics from the chosen domains (and, when
   the movie starts from a training or a deck, their concepts/slides). Access-filtered, as the person.
3. **Script.** `cortex_movie_script`: chapters → shots, each with narration (spoken, 1–3 sentences, cited) and a
   visual brief (what to show and how: clip/still/slide/screen). Checked like decks (`problems`, grounding re-ask).
   The visual prompts are then extended with the official Wan prompt extension (built in the lab, Gemma 4).
4. **Review.** The track appears. The person edits narration, swaps shot kinds, reorders, inserts, deletes, asks the
   Assistant for changes (proposals accepted per shot, as in Decks). Nothing has touched the GPU yet.
5. **Render.** Narration first (tts_server, all shots, seconds), which fixes durations. Then stills for every clip shot
   (seconds each) so the person sees the look before paying for motion. Then clips, preview quality, one job per shot,
   progress per shot on the track. A shot can be retaken (new seed or prompt) alone.
6. **Assemble.** ffmpeg: per-shot timing from narration, transitions, subtitles (SRT from the narration text, free),
   optional music bed, the AI-generated end card (D6), interpolation and upscaling at the final tier. One versioned MP4.
7. **Deliver.** Play in the tab, download, keep in the project; optionally submit the script (with shot thumbnails
   and the citations) to a domain through the review lifecycle, so the movie is found and cited by the Assistants.

### 3.4 The Movie channel in Cortex (what must exist, by the codebase's own checklist)

Backend `cortex/movies/`: `app.py` `Services` + `mount()` like decks; router `/api/movies`; socket events
`movie_progress`, `movie_shot`, `movie_done`, `movie_error`, `movie_assistant_*`; presets `cortex_movie_brief`,
`_script`, `_shot_prompt`, `_router`, prompts under `cortex/movies/prompts/`; `assistant_models.BUILTIN` row `movie`;
audit labels; retention; `delete_project` hook; settings (quality tier defaults, engine URL/token, GPU lease on/off);
`speech_status`-style health row for the engine. Jobs are persisted JSON records (the Decks `Generations` pattern),
resumable: on restart, "rendering" records with no task are marked interrupted and can be resumed shot by shot.

Frontend: `static/js/movies/` (`movies.js`, `assistant.js`, `views/track.js`, `views/shot.js`, `views/brief.js`,
`views/player.js`), `static/css/movies.css`, a side-menu entry (Workspace › Movies, listing the project's movies), a
middle-pane tab (`openTab('movie:<id>')`), the Assistant tab (`rightPanel.add('movie', …)`), menu entries in
`static/menu.json`, strings in `pt.js`, manual pages, Playwright suite `tests/ui/movies.mjs`, mockups
`design-mockups/movie.html` + `movie-plan.md` (Red Accent tokens only).

The **track view**: a horizontal strip of shot cards (thumbnail, duration, kind badge, status), chapters as groups,
a narration lane under each card, a playhead and a preview player above. Operations: select, drag to reorder, insert
(from text, from a still, from a deck slide, from a file), delete, split, merge two shots' narration, retake visual,
change kind, edit narration, set transition. This is the Decks slide list crossed with the Recording timeline; it is
built from DOM elements (cards), not the canvas, so it stays accessible and translatable.

---

## 4. GPU: when things run and who holds the card

Measured tenants on the 4090 (Cortex register #23 and this lab):

| Tenant | Resident | Can release? |
|---|---|---|
| llama-vision (gemma-4 + bge-m3 + ma2) | 8.7 GB | **no route today** (llama.cpp router) — see below |
| embeddings-server | 4.1 GB | yes: idle release (`/admin/gpu-memory`), models stay loaded; ~0.9 GB given back |
| ingestion-server | 3.5 GB | yes: idle release; ~0.9 GB given back |
| stt_server | 2.3 GB | yes: `/admin/release` lease, model to host RAM, back in 0.25 s |
| tts_server | 0.9 GB | yes: `/admin/release` lease |
| cortex | ~0 (CPU transcription) | — |
| **movie_engine** | **0 at idle; 11–20 GB during a job** (still 14.9, 480p clip 14.9, 720p clip 18.6, 720p I2V 19.9) | by design |

Resident total without a movie: 19.5 GB. A render needs 15–20 GB. So a render cannot overlap the resident set: the job
must run inside a lease.

**GPU lease for a render (Cortex-side `movies/gpu_lease.py`, same shape as `kbase/speech_lease.py`):**

1. Narration is produced first, while tts is loaded. Then the job asks, in parallel: stt and tts `POST /admin/release`
   (lease = job estimate, renewed every 240 s), embeddings and ingestion an explicit release (a `POST /admin/gpu-memory/release`
   to add next to the idle-release route), and the local model to unload (needs a route: llama.cpp's router has
   `/models/unload`-style control in recent builds — **to verify on the running image**; if absent, add a thin admin
   route to agent_server that asks the router to unload gemma-4 and reloads it after). The Assistant pane shows
   "Model paused: rendering a movie" the way speech shows "Speech paused: ingesting documents".
2. The job waits until `movie_engine /admin/state` reports enough free VRAM (it reads `nvidia-smi`; under WSL
   `cudaMemGetInfo` is not device-wide), with a timeout that fails the job cleanly instead of paging.
3. Shots render one by one; progress to the page; each finished shot is a file, so a cancel or crash loses at most
   one shot.
4. On the last shot (or cancel/failure) the lease ends: loads are requested, llama-vision reloads, speech reloads.
   A request that arrives during the lease reloads the model lazily, as today, and the movie job sees the slowdown
   rather than an error (measured: both sharing the card run 1.5–2x slower, not collapsing).

**Scheduling policy (settings):** `movie_gpu_lease` (default on), `movie_render_window` (optional: only start renders
between HH:MM–HH:MM, e.g. lunch/night, so working hours keep the Assistants fast), `movie_max_shots_per_job`. Previews
of stills (seconds) are allowed outside the window because they fit beside the resident set at Q8 14.9 GB only when
the card has ≥16 GB free — i.e. not with llama-vision up; so stills also use the lease unless the window is open. The
honest default: **everything GPU-heavy runs inside the lease, and the UI shows the queue position and the estimate.**

**Estimates shown to the person** (measured, 4-step, Q6_K): still ~15 s, 480p clip 90 s, 720p clip 3.8 min, 720p
image-to-video 4.8 min, narration seconds, assembly seconds. A 5-minute video with 12 clip shots at 480p preview
≈ 20 min; final 720p ≈ 50 min; interpolation + upscaling add a few minutes.

---

## 5. Post-processing helpers (inside the engine)

| Helper | Tool | Use | Verification |
|---|---|---|---|
| Frame interpolation | RIFE (ncnn or PyTorch) | 16 → 32/48 fps; slow-motion to stretch a shot to its narration | frame count × factor, PSNR of kept frames unchanged, visual check |
| Upscaling | Real-ESRGAN (fast, per-frame) first; SeedVR2 (video-aware) evaluated for the final tier | 480p preview → 1080p; 720p final → 1080p | resolution, temporal flicker check on a static shot |
| Ken Burns | ffmpeg zoompan | stills and slide shots over long narration | duration matches narration |
| Assembly | ffmpeg concat/xfade, subtitles, audio mux, loudness normalisation | the versioned MP4 + SRT | ffprobe streams, duration = sum of shots |
| AI marking | end card + container metadata | D6 | present in every export |

All run on the GPU inside the lease (interpolation/upscaling) or on the CPU (assembly), as engine job kinds.

---

## 6. Later (not release 1)

- **Presenter shots** through avatar_server (Ditto): a reference face + the narration audio → a lip-synced presenter
  clip, real-time cost. Needs an offline render route in avatar_server (it is a streaming server today).
- **Continuous long takes**: Wan VACE overlap conditioning for invisible seams; a subject LoRA for recurring characters.
- **Speech-to-video** (Wan S2V) for styled or full-body presenters.
- **pt-PT voice**: Kokoro has none; evaluate alternatives in tts_server.
- **Q8_0 at 720p** for a "best" tier (fits under cudaMallocAsync without the LoRA; 61 min per shot).

---

## 7. Phases, each with its proof

| Phase | Deliverable | Proof |
|---|---|---|
| 0 | `assets/movie_engine`: the lab as a service (Dockerfile, compose, bearer auth, jobs API, 0 GB idle, CI-ready layout) | contract tests on the fox prompts; `nvidia-smi` idle after each job |
| 1 | Helpers in the engine: interpolate, upscale, Ken Burns, assemble, subtitles, AI end card | the fox clips through the full post chain; measured times |
| 2 | Cortex backend: store, brief/script presets, TTS capture to files, job runner with persistence/resume, GPU lease (incl. the llama-vision unload route), notifications, audit, settings, Model Health row | a movie made from a real domain end to end by `tests/movies_check.py`; a lease measured with `nvidia-smi` before/during/after |
| 3 | Mockups (`design-mockups/movie.html`, both themes, 3 widths) → approval → frontend: track editor, Assistant tab, menus, i18n, manual, Playwright suite | screenshots in both themes; suite green; a person makes a movie without touching a terminal |
| 4 | Links: training → movie, deck slides as shots, publish script via review | a training becomes a movie; the movie is cited by the KB Assistant |
| 5 | Presenter via avatar_server; VACE; S2V | as above |

Order within phases follows the lab's working rule: measure, then build; every stage verified on its real output.

---

## 8. Open points to confirm on the live systems before Phase 2

- llama.cpp router: is there an unload/load control for a model id on the running `llama-vision` image? (Needed for
  the lease. Fallback: a lease that leaves gemma-4 resident and renders at 480p/Q6_K only, 14.9 GB — fits beside
  8.7 + the released rest. Measured free with everything else released: ~15 GB. So preview tier works even without it;
  720p does not.)
- tts_server: synthesis to a file (today only socket.io streaming; Cortex would be the socket client, or a small HTTP
  route is added to tts_server).
- avatar_server: offline render route (release 2).
- Lightning LoRA licence; SeedVR2 vs Real-ESRGAN choice by measurement.
