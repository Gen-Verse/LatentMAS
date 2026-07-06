#!/usr/bin/env bash
# Firewall CI wrapper (strategy.md §6). Usage:
#   bash scripts/firewall_check.sh [--target=latent_coordination|mechanistic_disentangle|all]
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET="all"
for arg in "$@"; do
  case "$arg" in
    --target=*) TARGET="${arg#--target=}" ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

exec python scripts/firewall_check.py --target "$TARGET"
