#!/usr/bin/env bash
# Run the same preparation step locally and, later, in the site build workflow.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/prepare_weirdml_v3.py "$@"
bundle exec jekyll build
