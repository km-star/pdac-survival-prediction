# push_to_github.ps1
# Run from inside the pdac-survival-prediction\ folder
# Requires: Git for Windows + GitHub CLI (gh) authenticated
#
# Usage:
#   cd pdac-survival-prediction
#   .\push_to_github.ps1

$REPO_NAME = "pdac-survival-prediction"
$GITHUB_USER = "km-star"
$DESCRIPTION = "Multi-omic PDAC survival prediction (LightGBM+XGBoost, ACC=0.841). Reproduces Nature Cancer 2024 pipeline on TCGA-PAAD."

Write-Host "==> Initializing git repo..." -ForegroundColor Cyan
git init
git add .
git commit -m "Initial commit: PDAC multi-omic survival prediction pipeline (ACC=0.841)"

Write-Host "==> Creating GitHub repo and pushing..." -ForegroundColor Cyan

if (Get-Command gh -ErrorAction SilentlyContinue) {
    gh repo create "$GITHUB_USER/$REPO_NAME" `
        --public `
        --description $DESCRIPTION `
        --source=. `
        --push
    Write-Host "==> Done! Repo live at: https://github.com/$GITHUB_USER/$REPO_NAME" -ForegroundColor Green
} else {
    Write-Host "gh CLI not found. Run these commands manually after creating the repo at https://github.com/new:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  git remote add origin git@github.com:$GITHUB_USER/$REPO_NAME.git"
    Write-Host "  git branch -M main"
    Write-Host "  git push -u origin main"
}
