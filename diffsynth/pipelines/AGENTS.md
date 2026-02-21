# Pipelines

High-level inference APIs for image/video generation.

## PIPELINE FILES

| File | Model Family | Purpose |
|------|--------------|---------|
| `flux_image.py` | FLUX.1 | Text-to-image, ControlNet, IP-Adapter |
| `flux2_image.py` | FLUX.2 | Text-to-image (dev, Klein variants) |
| `qwen_image.py` | Qwen-Image | Text-to-image, editing, inpainting |
| `wan_video.py` | Wan | Text/video-to-video |
| `ltx2_audio_video.py` | LTX-2 | Text/audio-to-video with speech |
| `z_image.py` | Z-Image | Text-to-image, image editing |

## COMMON PATTERN

```python
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
import torch

vram_config = {
    "offload_dtype": torch.float8_e4m3fn,
    "offload_device": "cpu",
    "onload_dtype": torch.float8_e4m3fn,
    "onload_device": "cpu",
    "preparing_dtype": torch.float8_e4m3fn,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="...", origin_file_pattern="*.safetensors", **vram_config),
    ],
    vram_limit=15.5,
)
image = pipe(prompt="...", seed=42)
```

## WHERE TO LOOK

| Task | Location |
|------|----------|
| Add new pipeline | Copy existing, inherit `BasePipeline` |
| VRAM config | `ModelConfig(**vram_config)` with 7 dtype/device pairs |
| Model loading | `from_pretrained()` + `model_pool.fetch_model()` |
| Inference units | `self.units` list in Pipeline `__init__` |
| Training hooks | `in_iteration_models` tuple for gradient tracking |
| Base class | `../diffusion/base_pipeline.py` |
