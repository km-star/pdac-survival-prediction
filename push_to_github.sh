#!/usr/bin/env bash
# push_to_github.sh
# Run this once from inside the pdac-survival-prediction/ directory.
# Prerequisites: git installed, GitHub CLI (gh) authenticated OR SSH key set up.
#
# Usage:
#   cd pdac-survival-prediction
#   chmod +x push_to_github.sh
#   ./push_to_github.sh

set -e

REPO_NAME="pdac-survival-prediction"
GITHUB_USER="km-star"   # your GitHub username

echo "==> Initializing git repo..."
git init
git add .
git commit -m "Initial commit: PDAC multi-omic survival prediction pipeline (ACC=0.841)"

echo "==> Creating GitHub repo and pushing..."
# Option A: GitHub CLI (recommended — install at https://cli.github.com)
if command -v gh &> /dev/null; then
    gh repo create "${GITHUB_USER}/${REPO_NAME}" \
        --public \
        --description "Multi-omic PDAC survival prediction (LightGBM+XGBoost, ACC=0.841). Reproduces Nature Cancer 2024 pipeline on TCGA-PAAD." \
        --source=. \
        --push
    echo "==> Done! Repo live at: https://github.com/${GITHUB_USER}/${REPO_NAME}"
else
    # Option B: Manual SSH push
    echo "gh CLI not found. Creating remote and pushing manually..."
    echo "  1. Go to https://github.com/new and create repo: ${REPO_NAME}"
    echo "  2. Then run:"
    echo "     git remote add origin git@github.com:${GITHUB_USER}/${REPO_NAME}.git"
    echo "     git branch -M main"
    echo "     git push -u origin main"
fi
