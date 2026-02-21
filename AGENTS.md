# DiffSynth-Studio Knowledge Base

**Generated:** 2026-02-21
**Commit:** bdcaacf
**Branch:** feat/refactor

## OVERVIEW

Open-source Diffusion model engine for image/video generation and training. Supports FLUX, Qwen-Image, Wan-Video, LTX-2, Z-Image families. PyTorch-based with advanced VRAM management (layer-level disk offload, FP8 quantization).

## STRUCTURE

```
DiffSynth-Studio/
├── diffsynth/           # Core library package
│   ├── models/          # Model architectures (VAE, DiT, text encoders)
│   ├── pipelines/       # Inference APIs per model family
│   ├── core/            # VRAM mgmt, loaders, attention, data
│   ├── utils/           # LoRA, ControlNet, state dict converters
│   ├── configs/         # Model configuration registry
│   └── diffusion/       # Training modules, loss functions
├── examples/            # Usage scripts by model family
│   ├── flux*/           # FLUX.1/FLUX.2 inference + training
│   ├── qwen_image/      # Qwen-Image inference + training
│   ├── wanvideo/        # Wan video generation
│   ├── ltx2/            # LTX-2 audio-video
│   └── z_image/         # Z-Image text-to-image
├── docs/                # Bilingual Sphinx docs (en/zh)
└── models/              # Local model storage (gitignored)
```

## WHERE TO LOOK

| Task | Location |
|------|----------|
| Run inference | `examples/<model>/model_inference/` |
| Low-VRAM inference | `examples/<model>/model_inference_low_vram/` |
| Train model | `examples/<model>/model_training/` |
| Add new model architecture | `diffsynth/models/` |
| Add new pipeline | `diffsynth/pipelines/` |
| VRAM management | `diffsynth/core/vram/` |
| Load models | `diffsynth/core/loader/` |
| LoRA utilities | `diffsynth/utils/lora/` |
| ControlNet | `diffsynth/utils/controlnet/` |
| State dict conversion | `diffsynth/utils/state_dict_converters/` |
| Model configs | `diffsynth/configs/model_configs.py` |

## CONVENTIONS

- **Line length:** 119 chars (Ruff)
- **Python target:** 3.11
- **Max complexity:** 12 (mccabe)
- **Quotes:** Double quotes
- **Indent:** Spaces
- **Model format:** Prefer `.safetensors` (not .bin/.pth/.ckpt)

## ANTI-PATTERNS

- **DO NOT** modify training parameters marked as fixed in example scripts
- **DO NOT** enable FP8 quantization for Z-Image Turbo (degrades quality)
- **DO NOT** set `denoising_strength != 1` without `input_image` (Qwen-Image/Z-Image)
- **DO NOT** call models outside `onload_model_names` in pipeline building
- **DO NOT** manually release VRAM after unit calculation
- **DO NOT** remove redundant parameters in DDP training (use `--find_unused_parameters`)
- **DO NOT** record loss values during training (not indicative of quality)
- **DO NOT** exceed 512 prompt tokens for Qwen-Image models

## COMMANDS

```bash
# Install (uv + hatchling backend)
uv sync

# Run example inference
python examples/flux/model_inference/FLUX.1-dev.py

# Run training (shell script)
bash examples/qwen_image/model_training/lora/Qwen-Image.sh

# Build docs
cd docs/en && make html

# Lint
uv ruff check diffsynth/
uv ruff format diffsynth/

# Build package
uv build
```

## ENVIRONMENT VARIABLES

| Variable | Purpose |
|----------|---------|
| `MODELSCOPE_DOMAIN` | Set to `www.modelscope.ai` for non-China users |
| `DIFFSYNTH_DOWNLOAD_SOURCE` | Override model download source |

## NOTES

- **No CLI:** Library only. Import pipelines in Python scripts.
- **VRAM limits:** Set conservatively (e.g., 15.5G for 16G GPU)
- **LTX-2:** Training not yet supported
- **DDP:** Requires `--find_unused_parameters` flag
- **Primary maintainer:** [Artiprocher](https://github.com/Artiprocher)
