#!/usr/bin/env bash
set -euo pipefail

REMOTE="${1:-origin}"
LOCAL_BRANCH="${2:-preview}"
REMOTE_BRANCH="${3:-previewn=}"

echo "Pushing local branch '${LOCAL_BRANCH}' to remote '${REMOTE}' as '${REMOTE_BRANCH}'"
git fetch "${REMOTE}"
git checkout -B "${LOCAL_BRANCH}"
git add -A
git commit -m "feat(preview): frontend/backend MVP - frame skip slider and violation logging" || true
git push -u "${REMOTE}" "${LOCAL_BRANCH}:${REMOTE_BRANCH}"
