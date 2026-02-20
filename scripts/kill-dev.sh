#!/usr/bin/env bash
# Kill all SynApps dev server processes (frontend + backend)
set -euo pipefail

PORTS="3000,3001,3002,8000,8001"

pids=$(lsof -i :${PORTS} -t 2>/dev/null | sort -u || true)

if [ -z "$pids" ]; then
  echo "No dev servers running on ports ${PORTS}"
  exit 0
fi

echo "Killing processes on ports ${PORTS}:"
lsof -i :${PORTS} 2>/dev/null | grep LISTEN || true
echo ""

kill $pids 2>/dev/null || true
sleep 1

# Force-kill any survivors
remaining=$(lsof -i :${PORTS} -t 2>/dev/null | sort -u || true)
if [ -n "$remaining" ]; then
  echo "Force-killing stubborn processes..."
  kill -9 $remaining 2>/dev/null || true
  sleep 1
fi

echo "Done. All dev server ports are free."
