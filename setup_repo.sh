#!/usr/bin/env bash
# One-time: initialize git and publish to GitHub.
# Requires: git, and GitHub CLI (Arch: sudo pacman -S github-cli; then `gh auth login`)
set -euo pipefail
REPO_NAME="${1:-prism}"
VISIBILITY="${2:---private}"   # pass --public to make it public

git init -b main
git add .
git commit -m "Initial commit: PRISM mesh network simulator"
gh repo create "$REPO_NAME" "$VISIBILITY" --source=. --remote=origin --push
echo "✅ Published: $(gh repo view --json url -q .url)"
