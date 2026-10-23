import hmac
import io
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

STATIC = os.path.join(os.path.dirname(__file__), "static")
BACKEND = os.getenv("BACKEND", "azure")            # "azure" (hosted FLUX) or "local" (FLUX Kontext on a GPU)
ACCESS_CODE = os.getenv("APP_PASSWORD", "")        # optional: require a code to generate
PREDICTOR = os.getenv("PREDICTOR", "azure")       # current-age estimate: "azure" (luna), "local" (ViT) or "off"
MAX_UPLOAD = 8 * 1024 * 1024
AGE = int(os.getenv("FUTURE_AGE", "50"))         # default target age
AGES = [20, 30, 40, 50, 60, 70, 80, 90, 100]     # ages the user can pick from (100 is the "in heaven" frame)
state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if BACKEND == "local":
        from .editor import AgeEditor
        state["local"] = AgeEditor()
    else:
        from .azure_editor import AzureEditor
        state["azure"] = AzureEditor()
    if PREDICTOR == "azure":                   # current-age estimate with a vision model on Azure
        from .azure_age import AzureAgeEstimator
        state["estimator"] = AzureAgeEstimator()
    elif PREDICTOR == "local":                 # small local ViT instead (needs requirements-gpu.txt)
        import torch
        from .predictor import AgePredictor
        state["predictor"] = AgePredictor("cuda" if torch.cuda.is_available() else "cpu")
    yield
    for key in ("azure", "estimator"):
        if key in state:
            await state[key].close()
    state.clear()


app = FastAPI(title="Future Self Photo Booth", lifespan=lifespan)

# The page can also be hosted elsewhere (GitHub Pages) and call this API across origins.
CORS_ORIGINS = [o for o in os.getenv("CORS_ORIGINS", "").split(",") if o]
if CORS_ORIGINS:
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST"],
                       allow_headers=["X-Access-Code"])


def _check_code(code: str | None):
    if ACCESS_CODE and not hmac.compare_digest(code or "", ACCESS_CODE):
        raise HTTPException(401, "Wrong or missing access code.")


async def _read_image(file: UploadFile) -> Image.Image:
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Photo is larger than 8 MB.")
    try:
        return Image.open(io.BytesIO(data))
    except UnidentifiedImageError:
        raise HTTPException(400, "That file is not an image.")


@app.get("/config")
def config():
    return {"needs_code": bool(ACCESS_CODE), "age": AGE, "ages": AGES,
            "predictor": "estimator" in state or "predictor" in state}


@app.post("/predict")
async def predict(file: UploadFile = File(...), x_access_code: str | None = Header(None)):
    """Estimate the current age: {"face": bool, "age": int | None}."""
    _check_code(x_access_code)
    image = await _read_image(file)
    if "estimator" in state:
        from .azure_age import AgeEstimateError
        try:
            return await state["estimator"].estimate(image)
        except AgeEstimateError as e:
            raise HTTPException(502, str(e))
    if "predictor" in state:
        out = await run_in_threadpool(state["predictor"].predict, image)
        return {"face": True, "age": round(out["age"])}
    raise HTTPException(404, "Age estimate is not enabled on this server.")


@app.post("/age")
async def age(file: UploadFile = File(...), target_age: int = Form(..., ge=10, le=100),
              seed: int = Form(7), current_age: int | None = Form(None, ge=1, le=99),
              x_access_code: str | None = Header(None)):
    _check_code(x_access_code)
    image = await _read_image(file)
    if "azure" in state:
        from .azure_editor import AzureEditorError
        try:
            out = await state["azure"].edit(image, target_age, seed, current_age)
        except AzureEditorError as e:
            raise HTTPException(e.status, e.message)
    else:
        out = await run_in_threadpool(state["local"].edit, image, target_age, seed)
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=92)
    return Response(buf.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


# Serve the page and its files from the root, the same layout GitHub Pages uses. API routes above win.
app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
