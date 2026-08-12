#!/bin/sh
set -e

if [ ! -f /app/data/nsl-kdd-train.csv ] || [ ! -f /app/data/nsl-kdd-test.csv ]; then
  echo "[ml-optimizer] dataset missing — running download_data.py"
  python download_data.py
fi

exec python -m app.worker
