#!/bin/sh
set -e

# Default training toggle
TRAIN_ENABLED=${TRAIN_AT_START:-true}

if [ "$TRAIN_ENABLED" != "false" ]; then
  python scripts/train_all.py
fi

exec "$@"
