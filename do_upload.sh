#!/usr/bin/env bash
# Run this once you have a valid HuggingFace token with Write permissions.
# Usage: bash do_upload.sh hf_YOURTOKEN vectrayx

set -euo pipefail

TOKEN="${1:-}"
ORG="${2:-vectrayx}"

if [ -z "$TOKEN" ]; then
    echo "Usage: bash do_upload.sh hf_YOURTOKEN [org]"
    echo "Get a Write token at: https://huggingface.co/settings/tokens"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Verifying token ==="
RESULT=$(curl -s -H "Authorization: Bearer $TOKEN" https://huggingface.co/api/whoami)
if echo "$RESULT" | grep -q '"error"'; then
    echo "ERROR: Token invalid — $RESULT"
    echo "Go to https://huggingface.co/settings/tokens and generate a new Write token."
    exit 1
fi
USERNAME=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['name'])")
echo "Logged in as: $USERNAME"
echo "Uploading to org: $ORG"
echo ""

echo "=== Uploading datasets (fast, ~5MB) ==="
HUGGING_FACE_HUB_TOKEN="$TOKEN" python3 "$SCRIPT_DIR/upload_to_hf.py" --org "$ORG" --datasets

echo ""
echo "=== Uploading code (fast, ~1MB) ==="
HUGGING_FACE_HUB_TOKEN="$TOKEN" python3 "$SCRIPT_DIR/upload_to_hf.py" --org "$ORG" --code

echo ""
echo "=== Uploading models (slow, ~500MB) ==="
HUGGING_FACE_HUB_TOKEN="$TOKEN" python3 "$SCRIPT_DIR/upload_to_hf.py" --org "$ORG" --models

echo ""
echo "=== Done! ==="
echo "Repos:"
echo "  https://huggingface.co/$ORG/vectrayx-nano"
echo "  https://huggingface.co/$ORG/vectrayx-bench"
echo "  https://huggingface.co/$ORG/vectrayx-paper-code"
