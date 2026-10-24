"""Age editor backed by a FLUX model hosted on Azure AI Foundry (BFL provider API)."""
import asyncio
import base64
import io
import os

import httpx
from PIL import Image

from .imaging import fit, match_colors

ENDPOINT = os.getenv("AZURE_ENDPOINT", "").rstrip("/")        # https://<resource>.services.ai.azure.com
API_KEY = os.getenv("AZURE_API_KEY", "")
MODEL = os.getenv("AZURE_MODEL", "FLUX.2-pro")
MAX_SIDE = int(os.getenv("MAX_SIDE", "832"))
MAX_PARALLEL = int(os.getenv("MAX_PARALLEL", "4"))
GUIDANCE = float(os.getenv("FLEX_GUIDANCE", "3.5"))   # FLUX.2-flex only: 1.5 to 10, higher follows the prompt more
STEPS = int(os.getenv("FLEX_STEPS", "30"))            # FLUX.2-flex only: max 50, fewer is faster

# Kept short on purpose: in side-by-side tests a long list of "keep the same X" items made the model redraw
# the face more. The age-specific line stops it from guessing (e.g. balding a 20-year-old at 30).
PROMPT = ("{start} {details} "
          "Keep the exact same skin tone and complexion as in the photo; do not lighten or whiten the skin. "
          "Keep the same ethnicity, facial features and facial hair (beard, mustache). "
          "Keep the same expression; do not add a smile. "
          "Keep the head in exactly the same position, size and angle so the photos line up. "
          "Keep everything else identical: face structure, hairstyle, clothing, background, lighting and framing.")

# The frames are played back as one continuous movie, so the signs of age must grow steadily from frame to frame
# (gray hair, wrinkles) and never jump back. Each band names one step on that ladder. Until the 60s the hairline
# and volume are locked, because the model otherwise invents balding.
AGE_DETAILS = [
    (34, "Subtle changes that make the face look slightly older and more mature, never younger. "
         "Hair stays exactly the same: same hairline, thickness, volume and length, no hair loss, no gray."),
    (39, "Slightly more mature face, very faint lines near the eyes. Same full hair and hairline, no gray yet."),
    (44, "Faint lines near the eyes and mouth. Same full hair and hairline, no balding, a few gray strands at most."),
    (49, "Light forehead lines and crow's feet. Same hairstyle and hairline, a little gray at the temples."),
    (54, "Clear forehead lines, crow's feet and smile lines. Same hairstyle and hairline, gray at the temples "
         "and some gray through the hair."),
    (59, "Defined wrinkles, slightly softer jawline. Same hairstyle, hair about one third gray."),
    (64, "Visible aging with fine wrinkles and softer skin. Same hairstyle, hair salt-and-pepper, about half gray."),
    (69, "Visible aging with wrinkles and softer skin. Same hairstyle, hair mostly gray."),
    (74, "Older face with deeper wrinkles and a few age spots. Same hairstyle, hair gray to white, slightly thinner."),
    (84, "Elderly: deep wrinkles, age spots, thinner skin. Same hairstyle, white hair, slightly thinner."),
    (200, "Very elderly: deep wrinkles, age spots, thin skin. Same hairstyle, thin white hair."),
]


HEAVEN_AGE = 100
# The 100 frame is a playful "in heaven" scene, so here the background and clothes change on purpose.
HEAVEN_PROMPT = (
    "Make this exact person look 100 years old and place them in heaven. "
    "Replace the background with soft glowing white clouds and warm golden light rays, add a subtle golden halo "
    "above the head, and dress them in simple white robes. "
    "Same person: identical face structure, ethnicity and skin tone; do not lighten or whiten the skin. "
    "Very old at 100: deep wrinkles, a few faint age spots, thin white hair in the same hairstyle, white facial hair where they "
    "have facial hair. Peaceful, gentle expression. Keep the same head pose and framing."
)


def _start(age: int, current_age: int | None) -> str:
    """Opening sentence. Knowing the current age lets the model size the change (22 -> 30 is only 8 years)."""
    if current_age is None or current_age >= age:
        return f"Make this exact person look {age} years old."
    gap = age - current_age
    if gap <= 10:
        size = (f"Only {gap} years pass, so the change is small and realistic: the face matures a little "
                "and stays clearly recognizable.")
    elif gap <= 25:
        size = f"{gap} years pass, so show moderate, realistic aging."
    else:
        size = f"{gap} years pass, so show strong but realistic aging."
    return (f"This person is about {current_age} years old in the photo. Show the same person {gap} years later, "
            f"at age {age}. {size}")


def build_prompt(age: int, current_age: int | None = None) -> str:
    if age >= HEAVEN_AGE:
        now = f"This person is about {current_age} years old in the photo. " if current_age else ""
        return now + HEAVEN_PROMPT
    details = next(d for limit, d in AGE_DETAILS if age <= limit)
    return PROMPT.format(start=_start(age, current_age), details=details)


class AzureEditorError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


class AzureEditor:
    def __init__(self):
        if not ENDPOINT or not API_KEY:
            raise RuntimeError("Set AZURE_ENDPOINT and AZURE_API_KEY to use the Azure backend")
        slug = MODEL.lower().replace(".", "-")   # FLUX.2-pro -> flux-2-pro
        self.url = f"{ENDPOINT}/providers/blackforestlabs/v1/{slug}?api-version=preview"
        self.client = httpx.AsyncClient(timeout=150)
        self.slots = asyncio.Semaphore(MAX_PARALLEL)

    async def edit(self, image: Image.Image, target_age: int, seed: int = 7,
                   current_age: int | None = None) -> Image.Image:
        img = fit(image, MAX_SIDE)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=95)
        body = {"prompt": build_prompt(target_age, current_age), "model": MODEL, "n": 1, "seed": seed,
                "input_image": base64.b64encode(buf.getvalue()).decode(),
                "width": img.width, "height": img.height}
        if "flex" in MODEL.lower():
            body.update(guidance=GUIDANCE, steps=STEPS)
        async with self.slots:
            r = await self.client.post(self.url, json=body, headers={"Authorization": f"Bearer {API_KEY}"})
        if r.status_code != 200:
            if r.status_code in (400, 422):
                raise AzureEditorError(422, "The image service declined this photo. Try a clear, front-facing photo.")
            if r.status_code == 429:
                raise AzureEditorError(429, "Too many requests right now. Wait a few seconds and try again.")
            raise AzureEditorError(502, f"The image service returned an error ({r.status_code}).")
        out = Image.open(io.BytesIO(base64.b64decode(r.json()["data"][0]["b64_json"]))).convert("RGB")
        return match_colors(out, img, same_scene=target_age < HEAVEN_AGE)

    async def close(self):
        await self.client.aclose()
