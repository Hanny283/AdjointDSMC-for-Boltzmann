#!/usr/bin/env bash
# ============================================================================
# pack_for_jupyterhub.sh
#
# Creates a clean deployment zip of the repository for uploading to JupyterHub.
# Large / generated directories are excluded automatically.
#
# Usage:
#   bash pack_for_jupyterhub.sh              # creates dsmc-shape-opt.zip
#   bash pack_for_jupyterhub.sh my-name.zip  # custom output filename
# ============================================================================

set -euo pipefail

ZIP_NAME="${1:-dsmc-shape-opt.zip}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_DIR="$(mktemp -d)"
PACK_DIR="$TMP_DIR/dsmc-shape-opt"

echo "=========================================="
echo "  Packing DSMC shape-optimisation code"
echo "=========================================="
echo "Source : $REPO_ROOT"
echo "Output : $ZIP_NAME"
echo ""

# -- Copy repository, excluding unwanted paths --------------------------------
rsync -a \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='venv' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='optimization_results' \
  --exclude='batch_results' \
  --exclude='batch_figures' \
  --exclude='hq_final' \
  --exclude='smoke_test' \
  --exclude='*.mp4' \
  --exclude='*.egg-info' \
  --exclude='.DS_Store' \
  "$REPO_ROOT/" "$PACK_DIR/"

# -- Create zip ---------------------------------------------------------------
cd "$TMP_DIR"
zip -r "$REPO_ROOT/$ZIP_NAME" dsmc-shape-opt/ -x "*.DS_Store"
cd "$REPO_ROOT"

# -- Cleanup ------------------------------------------------------------------
rm -rf "$TMP_DIR"

SIZE=$(du -sh "$ZIP_NAME" | cut -f1)
echo ""
echo "Created: $ZIP_NAME  ($SIZE)"
echo ""
echo "=========================================="
echo "  Next steps"
echo "=========================================="
echo ""
echo "Option A — Upload via JupyterHub browser:"
echo "  1. Go to JupyterHub → File Browser → Upload"
echo "  2. Upload $ZIP_NAME"
echo "  3. Open a Terminal in JupyterHub and run:"
echo "       unzip ~/dsmc-shape-opt.zip -d ~/"
echo "       # repo is now at ~/dsmc-shape-opt/"
echo ""
echo "Option B — Git clone (if repo is on GitHub/GitLab):"
echo "  Push this repo to your remote first:"
echo "       git remote add origin https://github.com/YOU/YOUR-REPO.git"
echo "       git push -u origin main"
echo "  Then on JupyterHub Terminal:"
echo "       git clone https://github.com/YOU/YOUR-REPO.git ~/dsmc-shape-opt"
echo ""
echo "Option C — SCP (if you have SSH access to the server):"
echo "       scp $ZIP_NAME user@server:~/"
echo "       ssh user@server 'unzip ~/dsmc-shape-opt.zip -d ~/'"
echo ""
echo "After extraction, open notebooks/jupyterhub_run.ipynb in JupyterHub"
echo "and follow the cells top-to-bottom."
echo ""
echo "  Quick batch run from Terminal:"
echo "       cd ~/dsmc-shape-opt"
echo "       nohup python experiments/batch_run.py \\"
echo "           --n-runs 30 --maxiter 50 --workers -1 \\"
echo "           > batch.log 2>&1 &"
echo "       tail -f batch.log"
echo ""
echo "  Analyse results:"
echo "       python experiments/compare_results.py --results-dir batch_results"
echo "=========================================="
