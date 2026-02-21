# Utils Knowledge Base

Utility modules for LoRA, ControlNet, state dict conversion, and sequence parallelism.

## STRUCTURE

```
utils/
├── lora/                    # LoRA manipulation utilities
│   ├── merge.py            # Merge multiple LoRAs into one
│   ├── reset_rank.py       # Adjust LoRA rank (compress/expand)
│   ├── flux.py             # FLUX-specific LoRA operations
│   └── general.py          # Generic LoRA utilities
├── controlnet/             # ControlNet support
│   ├── annotator.py        # Image annotation (Canny, Depth, etc.)
│   └── controlnet_input.py # Control input processing
├── state_dict_converters/  # Checkpoint format converters (~25 files)
│   ├── flux_*.py           # FLUX DiT, VAE, text encoders
│   ├── wan_video_*.py      # Wan Video components
│   ├── ltx2_*.py           # LTX-2 components
│   ├── z_image_*.py        # Z-Image components
│   └── qwen_image_*.py     # Qwen-Image components
└── xfuser/                 # Sequence parallelism
    └── xdit_context_parallel.py  # xDiT context parallel for long sequences
```

## WHERE TO LOOK

| Task | Module |
|------|--------|
| Merge LoRA weights | `lora/merge.py` |
| Reduce LoRA file size | `lora/reset_rank.py` |
| Load FLUX LoRA | `lora/flux.py` |
| Convert HuggingFace checkpoint | `state_dict_converters/<model>_*.py` |
| Add Canny/Depth control | `controlnet/annotator.py` |
| Multi-GPU sequence parallel | `xfuser/xdit_context_parallel.py` |

## PATTERNS

- **Converters:** One file per model component. Match filename to model config key.
- **LoRA merge:** Combine multiple LoRAs with adjustable weights before inference.
- **Rank reset:** Lower rank = smaller file, may reduce quality slightly.
- **xDiT:** Use for long video generation when single GPU memory insufficient.

## NOTES

- Converters normalize keys between HuggingFace and internal formats.
- FLUX LoRA uses special handling due to double-stream architecture.
- Annotators cache models after first load. Call `unload()` to free VRAM.
