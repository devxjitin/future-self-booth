"""Fast age estimation with a small ViT classifier (expected value over age buckets)."""
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "nateraw/vit-age-classifier"
# midpoint of each bucket the model outputs
BUCKET_MID = {"0-2": 1, "3-9": 6, "10-19": 14.5, "20-29": 24.5, "30-39": 34.5,
              "40-49": 44.5, "50-59": 54.5, "60-69": 64.5, "more than 70": 75}


class AgePredictor:
    def __init__(self, device: str):
        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.model = (AutoModelForImageClassification.from_pretrained(MODEL_ID, torch_dtype=self.dtype)
                      .to(device).eval())

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict:
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device, self.dtype)
        probs = self.model(**inputs).logits.softmax(-1)[0].float().cpu()
        labels = [self.model.config.id2label[i] for i in range(len(probs))]
        buckets = {l: round(p.item(), 4) for l, p in zip(labels, probs)}
        age = sum(BUCKET_MID[l] * p for l, p in buckets.items())
        return {"age": round(age, 1), "top_bucket": max(buckets, key=buckets.get), "buckets": buckets}
