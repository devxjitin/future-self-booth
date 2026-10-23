"""Current-age estimate from a photo, using a vision chat model deployed on Azure (default gpt-5.6-luna)."""
import base64
import io
import json
import os

import httpx
from PIL import Image

from .imaging import fit

ENDPOINT = os.getenv("AZURE_ENDPOINT", "").rstrip("/")
API_KEY = os.getenv("AZURE_API_KEY", "")
# In tests on a 22-year-old, gpt-5.6-luna answered 22-25 and gpt-6-luna 28-30, so 5.6 is the default.
MODEL = os.getenv("AGE_MODEL", "gpt-5.6-luna")
API_VERSION = os.getenv("AGE_API_VERSION", "2025-04-01-preview")

SYSTEM = ("You estimate the apparent age of the main face in a photo for a fun photo booth. "
          "Reply with JSON only: {\"age\": <integer>, \"face\": <true|false>}. "
          "Use face=false and age=0 when there is no clear human face.")


class AgeEstimateError(Exception):
    pass


class AzureAgeEstimator:
    def __init__(self):
        if not ENDPOINT or not API_KEY:
            raise RuntimeError("Set AZURE_ENDPOINT and AZURE_API_KEY to use the Azure age estimate")
        self.url = f"{ENDPOINT}/openai/deployments/{MODEL}/chat/completions?api-version={API_VERSION}"
        self.client = httpx.AsyncClient(timeout=60)

    async def estimate(self, image: Image.Image) -> dict:
        img = fit(image, 512)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        body = {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": [
                    {"type": "text", "text": "Estimate the apparent age."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]},
            ],
            "max_completion_tokens": 300,
            "response_format": {"type": "json_object"},
        }
        r = await self.client.post(self.url, json=body, headers={"api-key": API_KEY})
        if r.status_code != 200:
            raise AgeEstimateError(f"Age service returned an error ({r.status_code}).")
        try:
            out = json.loads(r.json()["choices"][0]["message"]["content"])
            face, age = bool(out.get("face")), int(out.get("age", 0))
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            raise AgeEstimateError("Age service gave an answer that could not be read.")
        return {"face": face, "age": max(1, min(99, age)) if face else None}

    async def close(self):
        await self.client.aclose()
