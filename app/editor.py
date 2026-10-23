"""FLUX.1 Kontext [dev] age editor. Kept loaded in GPU memory; speed knobs come from env vars."""
import os
import threading

import torch
from diffusers import FluxKontextPipeline
from PIL import Image

from .imaging import fit

MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
STEPS = int(os.getenv("KONTEXT_STEPS", "20"))
MAX_SIDE = int(os.getenv("MAX_SIDE", "768"))
GUIDANCE = float(os.getenv("KONTEXT_GUIDANCE", "2.5"))

PROMPT = ("Make this person look {age} years old. Keep the same identity, facial structure, "
          "expression, pose, clothing and background.")


def _fit(image: Image.Image) -> Image.Image:
    return fit(image, MAX_SIDE)


class AgeEditor:
    def __init__(self):
        self.lock = threading.Lock()  # one generation at a time on the GPU
        self.pipe = FluxKontextPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        if os.getenv("COMPILE") == "1":
            self.pipe.transformer = torch.compile(self.pipe.transformer, mode="max-autotune-no-cudagraphs")
        self.edit(Image.new("RGB", (MAX_SIDE, MAX_SIDE), "gray"), 40)  # warm-up (also triggers compile)

    @torch.inference_mode()
    def edit(self, image: Image.Image, target_age: int, seed: int = 0) -> Image.Image:
        img = _fit(image.convert("RGB"))
        with self.lock:
            gen = torch.Generator("cuda").manual_seed(seed)
            return self.pipe(image=img, prompt=PROMPT.format(age=target_age), width=img.width,
                             height=img.height, guidance_scale=GUIDANCE, num_inference_steps=STEPS,
                             generator=gen).images[0]
