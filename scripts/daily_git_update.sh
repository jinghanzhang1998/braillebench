#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

git rev-parse --is-inside-work-tree >/dev/null

if [[ -z "$(git status --porcelain)" ]]; then
  echo "No code changes to commit."
  exit 0
fi

git add -A

if git diff --cached --quiet; then
  echo "No staged changes after applying .gitignore."
  exit 0
fi

git commit -m "daily code update $(date +%Y-%m-%d)"

if git remote get-url origin >/dev/null 2>&1; then
  git push origin HEAD
else
  echo "No remote origin configured; committed locally only."
fi
