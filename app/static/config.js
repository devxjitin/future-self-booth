// Where the page finds its API. Locally the FastAPI server serves both, so nothing is set.
// On GitHub Pages the API runs on AWS Lambda; deploy/lambda.ps1 prints the URL to put here.
if (!/^(localhost|127\.0\.0\.1)$/.test(location.hostname)) {
  window.API_BASE = "https://ihrfijyiartejmpjrhr24zb6vu0mlmrd.lambda-url.us-east-1.on.aws";
}
