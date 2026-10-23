# Deploys the app to a single GPU EC2 instance via CloudFormation.
# Prereqs: `aws login` done, and the HF token stored once with:
#   aws ssm put-parameter --name /age-prediction/hf-token --type SecureString --value <hf_token>
# (the token's account must have accepted the FLUX.1-Kontext-dev license on Hugging Face)
param(
    [string]$Region = "us-east-1",
    [string]$InstanceType = "g6e.xlarge",
    [string]$StackName = "age-prediction"
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent

$account = aws sts get-caller-identity --query Account --output text
$bucket = "age-prediction-code-$account-$Region"
$ErrorActionPreference = "Continue"   # PS 5.1 treats native stderr as a terminating error otherwise
aws s3api head-bucket --bucket $bucket 2>$null
$bucketExists = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = "Stop"
if (-not $bucketExists) { aws s3 mb "s3://$bucket" --region $Region }

# bsdtar writes a zip with forward-slash paths (Compress-Archive on PS 5.1 does not)
$zip = Join-Path $env:TEMP "app.zip"
if (Test-Path $zip) { Remove-Item $zip }
tar -a -c -f $zip -C $root --exclude=__pycache__ app requirements.txt requirements-gpu.txt
aws s3 cp $zip "s3://$bucket/app.zip" --region $Region

$vpc = aws ec2 describe-vpcs --filters Name=is-default,Values=true --query "Vpcs[0].VpcId" --output text --region $Region
$subnet = aws ec2 describe-subnets --filters Name=vpc-id,Values=$vpc --query "Subnets[0].SubnetId" --output text --region $Region
$myIp = (Invoke-RestMethod https://checkip.amazonaws.com).Trim()

aws cloudformation deploy --stack-name $StackName --region $Region --capabilities CAPABILITY_IAM `
    --template-file (Join-Path $PSScriptRoot "stack.yaml") --parameter-overrides `
    InstanceType=$InstanceType VpcId=$vpc SubnetId=$subnet AllowedCidr="$myIp/32" CodeBucket=$bucket

aws cloudformation describe-stacks --stack-name $StackName --region $Region --query "Stacks[0].Outputs" --output table
Write-Host "First boot downloads ~35GB of weights; watch with: aws ssm start-session --target <InstanceId>  then  journalctl -u age -f"
