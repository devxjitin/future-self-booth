# Future Self Photo Booth

A just-for-fun web app. The camera takes a photo and one button, "Grow my future self", generates a batch of frames (every 5 years from about 30 to 60, then 70, 80, 90, plus 100 "in heaven" with clouds, golden light, halo and white robes). The frames play back as a growing movie on a canvas: each face morphs into the next (the prompt locks head position so the photos line up), the age counter runs up along with the blend, with a dot timeline, Replay, and "Save this frame". Frames start playing as they arrive. Ages too close to the guessed current age are skipped. The page has an animated AI background (drifting aurora, neural network with data pulses, moving grid floor), always fits the screen with no scrolling, and the camera is the only way to add a photo.

The images come from a FLUX model hosted on Azure AI Foundry (`FLUX.2-pro` by default). They are AI guesses, not forecasts, and identity is not always preserved (eye colour, face shape and expression can drift).

After the photo is taken, `gpt-5.6-luna` guesses the current age ("You look 23–27 today", shown as a range because the guess is only accurate to a few years). The guessed age is also sent with the image request, so the prompt says how many years pass (for example "about 22 now, show the same person 8 years later") and keeps a small gap small. If no face is found, the app asks for a retake.

How the edit is steered (`app/azure_editor.py`, `app/imaging.py`):
- The prompt adds age-specific details, so younger targets keep the same hair (no balding or gray before the 40s) and older ones gray naturally.
- It asks to keep skin tone, ethnicity, facial hair and expression. The model still tends to lighten skin, so after generation the colours are matched back to the input photo: once for the whole frame and once more for the face area in the centre.

## Run it
```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```
Set these environment variables (never commit them):

| Variable | Meaning |
|---|---|
| `AZURE_ENDPOINT` | `https://<resource>.services.ai.azure.com` |
| `AZURE_API_KEY` | key of that Foundry resource |
| `AZURE_MODEL` | optional, default `FLUX.2-pro` (`FLUX.2-flex` also works) |
| `FLEX_GUIDANCE`, `FLEX_STEPS` | optional, only used with `FLUX.2-flex`, defaults `3.5` and `30` |
| `FUTURE_AGE` | optional, the age selected at start, default `50` |
| `PREDICTOR` | optional current-age guess: `azure` (default, uses `AGE_MODEL`), `local` (ViT, needs `requirements-gpu.txt`) or `off` |
| `AGE_MODEL` | optional, the Azure vision deployment for the current-age guess, default `gpt-5.6-luna` |
| `APP_PASSWORD` | optional access code required to generate (good for classroom demos) |

```bash
.venv\Scripts\uvicorn app.main:app --port 8000
```
Open http://127.0.0.1:8000 in the browser. Do not open `index.html` as a file: the page then cannot reach the server and shows "Failed to fetch".

## Deploy (GitHub Pages + AWS Lambda)
The page is static and lives on GitHub Pages at https://futureself.devxjitin.dev. The API (which holds the Azure key) runs on AWS Lambda behind a public HTTPS Function URL.

1. API: set `AZURE_ENDPOINT` and `AZURE_API_KEY` in the shell, `aws login`, then run `deploy/lambda.ps1`. It packages the app, creates or updates the `future-self-booth-api` function, and writes its URL into `app/static/config.js`. The API only accepts browser calls from the Pages domain and localhost (CORS).
2. Page: push to `main`. `.github/workflows/pages.yml` publishes `app/static` to Pages.
3. Custom domain: at the DNS provider of `devxjitin.dev` (name.com), add a `CNAME` record `futureself` pointing to `devxjitin.github.io`. Then turn on "Enforce HTTPS" in the repo's Pages settings once the certificate is ready.

The API is open (no access code), so anyone with the page can generate images on the Azure bill. Azure's rate limit (about 15 requests per minute for FLUX.2-pro) is the only cap.

## Using it with students
- Use photos of consenting adults (for example teachers) or AI-generated faces, not students' photos or celebrity photos.
- Photos are sent to Azure, so check your school's data policy first.
- Set `APP_PASSWORD` so only the teacher's session can generate images, and watch Azure costs, since every frame is a paid image.

## Other backends
- `BACKEND=local` runs FLUX.1 Kontext [dev] on your own GPU (24GB+ VRAM, `requirements-gpu.txt`). `deploy/` is a CloudFormation stack for that, on one g6e instance. It needs a GPU quota on your AWS account and a Hugging Face token with the model license accepted. Kontext [dev] weights are non-commercial.
- `ENABLE_PREDICTOR=1` adds a `/predict` endpoint that guesses the current age with a small ViT (needs `requirements-gpu.txt`). The UI does not use it yet.

## Notes
- A batch is 11 FLUX calls, 4 at a time (about 50 to 70 seconds in all, within the 15 requests per minute limit); a frame that hits a rate limit is retried. All frames of a batch share one seed so the face stays consistent, "Grow again" uses a new seed.
- The camera needs `localhost` or HTTPS.
- Idea for better identity: Cradle2Cane (github.com/byliutao/Cradle2Cane, non-commercial licence) is a fast SDXL-Turbo based aging model that takes a single photo. It is not integrated yet.
