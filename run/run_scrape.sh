#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

python inside_scrape.py >> logs/inside_scrape.cron.log 2>&1
