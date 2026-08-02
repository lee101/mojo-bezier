#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/dist"
mojo build --emit shared-lib "$ROOT/src/capi.mojo" -o "$ROOT/dist/libmojo-bezier.so"
