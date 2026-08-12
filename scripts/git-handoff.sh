#!/usr/bin/env bash
# =============================================================================
# BlackICE-Mesh — Version Control Handoff
# Initializes the new repository, severs ties with the old Adv-Guard history,
# and pushes to the BlackICE-Mesh remote.
#
# Usage:
#   bash scripts/git-handoff.sh          # init + commit + remote + push
#   bash scripts/git-handoff.sh --dry    # show commands without running them
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_URL="https://github.com/pd241008/BlackICE-Mesh.git"
COMMIT_MSG="INIT: BlackICE-Mesh polyglot microservices architecture"
DRY=false

if [[ "${1:-}" == "--dry" ]]; then
  DRY=true
fi

run() {
  if $DRY; then
    echo "[dry-run] $*"
  else
    echo "[run] $*"
    eval "$*"
  fi
}

cd "$REPO_ROOT"

if $DRY; then
  run "git init"
  run "git add ."
  run "git commit -m \"$COMMIT_MSG\""
  run "git remote add origin $REMOTE_URL"
  run "git branch -M main"
  run "git push -u origin main"
  echo "--- dry run complete ---"
  exit 0
fi

git init
git add .
git commit -m "$COMMIT_MSG"

if git remote get-url origin >/dev/null 2>&1; then
  echo "[skip] origin already configured: $(git remote get-url origin)"
else
  git remote add origin "$REMOTE_URL"
fi

git branch -M main
git push -u origin main

echo "=== BlackICE-Mesh pushed to $REMOTE_URL ==="
