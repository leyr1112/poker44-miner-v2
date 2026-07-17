#!/usr/bin/env bash
# Start the Poker44 v2 miner. Reads wallet/port config from .env.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f .env ]; then
  set -a; . ./.env; set +a
else
  echo "missing .env -- copy .env.example to .env and fill it in" >&2
  exit 1
fi

: "${POKER44_WALLET_NAME:?set POKER44_WALLET_NAME in .env}"
: "${POKER44_WALLET_HOTKEY:?set POKER44_WALLET_HOTKEY in .env}"

exec "${POKER44_PYTHON:-python3}" neurons/miner.py \
  --netuid "${POKER44_NETUID:-126}" \
  --wallet.name "$POKER44_WALLET_NAME" \
  --wallet.hotkey "$POKER44_WALLET_HOTKEY" \
  --subtensor.network "${POKER44_SUBTENSOR_NETWORK:-finney}" \
  --axon.port "${POKER44_AXON_PORT:-8091}" \
  --blacklist.force_validator_permit
