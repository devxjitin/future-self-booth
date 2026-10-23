# Deploys the API (FastAPI + Azure FLUX/luna) to AWS Lambda with a public HTTPS Function URL,
# then writes that URL into app/static/config.js for the GitHub Pages frontend.
# Needs: `aws login`, and AZURE_ENDPOINT / AZURE_API_KEY set in this shell (they go into the Lambda's
# encrypted environment, never into the repo).
param(
    [string]$Region = "us-east-1",
    [string]$Name = "future-self-booth-api",
    [string]$Origins = "https://ageprediction.devxjitin.dev,http://127.0.0.1:8000,http://localhost:8000"
)
$ErrorActionPreference = "Continue"   # PS 5.1 treats native stderr as errors; exit codes are checked instead
if (-not $env:AZURE_ENDPOINT -or -not $env:AZURE_API_KEY) { throw "Set AZURE_ENDPOINT and AZURE_API_KEY first." }
$root = Split-Path $PSScriptRoot -Parent
$build = Join-Path $root "build\lambda"
$zip = Join-Path $root "build\lambda.zip"

# 1. Package: Linux wheels for Python 3.12 plus the app code
if (Test-Path $build) { Remove-Item $build -Recurse -Force }
New-Item -ItemType Directory -Force $build | Out-Null
& (Join-Path $root ".venv\Scripts\python.exe") -m pip install -q -r (Join-Path $root "requirements-lambda.txt") `
    --platform manylinux2014_x86_64 --implementation cp --python-version 3.12 --only-binary=:all: --target $build
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
Copy-Item (Join-Path $root "app") (Join-Path $build "app") -Recurse
Get-ChildItem $build -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
if (Test-Path $zip) { Remove-Item $zip }
tar -a -c -f $zip -C $build *
if ($LASTEXITCODE -ne 0) { throw "zip failed" }

# 2. Execution role (logs only)
$roleName = "$Name-role"
$roleArn = aws iam get-role --role-name $roleName --query Role.Arn --output text 2>$null
if (-not $roleArn -or $LASTEXITCODE -ne 0) {
    $trust = Join-Path $env:TEMP "lambda-trust.json"
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' |
        Set-Content $trust -Encoding ascii
    $roleArn = aws iam create-role --role-name $roleName --assume-role-policy-document "file://$trust" --query Role.Arn --output text
    if ($LASTEXITCODE -ne 0) { throw "create-role failed" }
    aws iam attach-role-policy --role-name $roleName --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
    Start-Sleep -Seconds 10   # new roles take a moment before Lambda can assume them
}

# 3. Function (create or update). Env vars go through a temp file so the key never shows on a command line.
$envFile = Join-Path $env:TEMP "lambda-env.json"
@{ Variables = @{ AZURE_ENDPOINT = $env:AZURE_ENDPOINT; AZURE_API_KEY = $env:AZURE_API_KEY; CORS_ORIGINS = $Origins;
                  BACKEND = "azure"; PREDICTOR = "azure" } } | ConvertTo-Json -Compress | Set-Content $envFile -Encoding ascii
try {
    aws lambda get-function --function-name $Name --region $Region 2>$null | Out-Null
    $exists = ($LASTEXITCODE -eq 0)
    if ($exists) {
        aws lambda update-function-code --function-name $Name --zip-file "fileb://$zip" --region $Region | Out-Null
        aws lambda wait function-updated --function-name $Name --region $Region
        aws lambda update-function-configuration --function-name $Name --environment "file://$envFile" `
            --timeout 60 --memory-size 1024 --region $Region | Out-Null
    } else {
        aws lambda create-function --function-name $Name --runtime python3.12 --handler app.lambda_handler.handler `
            --role $roleArn --zip-file "fileb://$zip" --timeout 60 --memory-size 1024 `
            --environment "file://$envFile" --region $Region | Out-Null
    }
    if ($LASTEXITCODE -ne 0) { throw "creating/updating the function failed" }
    aws lambda wait function-updated --function-name $Name --region $Region
} finally {
    Remove-Item $envFile -ErrorAction SilentlyContinue
}

# 4. Public HTTPS Function URL (CORS is handled by the app itself)
$url = aws lambda get-function-url-config --function-name $Name --region $Region --query FunctionUrl --output text 2>$null
if (-not $url -or $LASTEXITCODE -ne 0) {
    $url = aws lambda create-function-url-config --function-name $Name --auth-type NONE --region $Region --query FunctionUrl --output text
    if ($LASTEXITCODE -ne 0) { throw "create-function-url-config failed" }
    aws lambda add-permission --function-name $Name --statement-id public-url --action lambda:InvokeFunctionUrl `
        --principal "*" --function-url-auth-type NONE --region $Region | Out-Null
    # Since late 2025 a public URL also needs InvokeFunction, limited here to calls that come through the URL
    aws lambda add-permission --function-name $Name --statement-id public-invoke --action lambda:InvokeFunction `
        --principal "*" --invoked-via-function-url --region $Region | Out-Null
}
$url = $url.TrimEnd("/")

# 5. Point the Pages frontend at it
$config = Join-Path $root "app\static\config.js"
(Get-Content $config -Raw) -replace 'window\.API_BASE = "[^"]*"', "window.API_BASE = `"$url`"" | Set-Content $config -Encoding utf8 -NoNewline
Write-Host "API URL: $url"
