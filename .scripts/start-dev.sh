#!/bin/bash

# Step 1: Clear ports (clean kill from last run)
kill-port 8000 3000

# Step 2: Run both backend + frontend with labels and coloring
concurrently \
  -n "🧠 API,🎨 UI" \
  -c "cyan.bold,magenta.bold" \
  "cd apps/orchestrator && uvicorn main:app --reload" \
  "cd apps/web-frontend && npm start"
