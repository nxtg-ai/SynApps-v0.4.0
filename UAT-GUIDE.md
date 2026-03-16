# SynApps v1.0 — Human UAT Guide

> **Tester**: Asif Waliuddin (Founder)
> **Date**: 2026-02-22
> **Purpose**: Hands-on UX evaluation. Automated tests cover correctness — this guide covers **feel**.

---

## How to Start

Two terminals in WSL:

```bash
# Terminal 1 — Backend
cd ~/projects/synapps
source .venv/bin/activate
PYTHONPATH=. uvicorn apps.orchestrator.main:app --reload --port 8000

# Terminal 2 — Frontend
cd ~/projects/synapps/apps/web-frontend
npm run dev
```

Open **http://localhost:3000**. Health check: http://localhost:8000/api/v1/health should return `{"status": "ok"}`.

For the 2Brain template, Ollama must be running in WSL with `llama3.1`:
```bash
ollama serve          # if not already running
ollama pull llama3.1  # ~4.7GB, one-time download
```

---

## Part 1: Workflow Creation UX

> Focus: How does it *feel* to go from zero to a working workflow?

### 1.1 First Impression (Dashboard)

Open the dashboard fresh. Take 10 seconds and just look.

- Does the layout make sense instantly? Or do you need to hunt for things?
- Is "Create New Workflow" the most prominent action? It should be.
- Do the Featured Templates give you a clear idea of what SynApps does?
- Does anything say "v0.4.0" or "Alpha"? (It shouldn't.)

### 1.2 Template Selection Flow

Click "Create New Workflow".

- How long does the modal take to appear? Does it feel instant?
- Are the 4 templates self-explanatory from their names and descriptions?
- When you click a template card, does the selection feel responsive? (Blue highlight)
- Is "Create Flow" clearly the next step, or do you hesitate?

### 1.3 Building from Scratch

Go back to dashboard, click "Create New Workflow", but this time close the template modal and go to `/editor` directly.

- Start with blank canvas (Start + End nodes only). Is it obvious how to add nodes?
- **Drag** a node from the palette onto the canvas. Does it land where expected?
- **Click** a node in the palette. Where does it appear? Is placement predictable?
- Connect Start → new node → End by dragging between handles. Does the connection snap feel natural, or do you fight with the handles?
- How many attempts does it take to connect your first edge? (1 = great, 3+ = friction)

### 1.4 Node Configuration

Double-click an LLM node to open its config.

- Do you immediately understand what each field does?
- Is the Provider dropdown obvious? Does switching to "ollama" feel smooth?
- Temperature slider — can you set precise values, or is it too coarse?
- When you close the modal, do your settings persist? (Reopen to check.)
- Try configuring a Code node — is the code textarea usable, or too small?

**Friction check**: Rate each node type's config modal 1-5 for clarity:
| Node Type | Clarity (1-5) | Notes |
|-----------|---------------|-------|
| LLM | | |
| Writer | | |
| Artist | | |
| Memory | | |
| Code | | |
| If/Else | | |
| Merge | | |
| For-Each | | |

---

## Part 2: Running a Workflow

### 2.1 Run Button Behavior

Save your workflow, then click "Run Workflow".

- Does the button change to "Running..." immediately?
- Does it STAY in "Running..." for the full execution? (It should. This was a recent fix.)
- Or does it flash back to "Run Workflow" before results appear? (That's a bug — report it.)

### 2.2 Execution Visualization

While a workflow runs, watch the canvas:

- Do active nodes glow? Is the glow visible or too subtle?
- Does the running node show a spinning indicator?
- Do edges show animated particles? Are they smooth or janky?
- When a node completes, does the green badge appear with a satisfying pop?
- Does the mini-output preview on completed nodes add value, or is it noise?

### 2.3 Output Clarity

After the run completes, check the Output Data panel:

- Is it obvious where the results are?
- Can you distinguish "Text Output" from "Raw Results"?
- If the run failed, is the error message helpful or cryptic?
- Does the output panel auto-scroll to show results, or do you need to scroll?

### 2.4 Error Recovery

Intentionally break something (bad API key, disconnect from Ollama) and run:

- Does the error message tell you what went wrong and which node failed?
- Can you fix the issue and re-run without refreshing the page?
- Does the canvas correctly show error badges on failed nodes?

---

## Part 3: Template Usability

### 3.1 Blog Post Writer

Select the Blog Post Writer template.

- 5 nodes appear: Start → Draft Writer → Store Draft → Refinement Writer → End
- Is the two-stage writing concept (draft → refine) clear from the node names?
- Run it with input: `{"text": "Write about the future of local AI"}`. Does the output feel like a refined blog post?

### 3.2 2Brain Inbox Triage (see Part 5 for deep-dive)

Quick check here: does the template load with all 5 nodes connected? Are there any stray nodes or missing edges?

### 3.3 Template Consistency

For each template, note:
- Does the node layout look intentional (clean vertical flow) or randomly placed?
- Are all edges connected? Any orphaned nodes?
- Does the workflow name pre-fill correctly in the toolbar?

---

## Part 4: Debug Output Clarity

### 4.1 Browser Console

Open DevTools (F12) → Console tab before running any workflow.

- Are there noisy warnings cluttering the console? (React dev warnings are OK)
- When a run fails, does the console show a clear error with stack trace?
- Any `Unhandled Promise Rejection` warnings? (Those are bugs.)
- WebSocket messages — are they visible in the Network tab as expected?

### 4.2 History Page

Navigate to `/history` after running a few workflows.

- Does each run show: status badge, flow name, timestamp?
- Can you tell at a glance which runs succeeded and which failed?
- Clicking a run — does it expand with useful detail?

---

## Part 5: 2Brain Inbox Triage — End-to-End Dogfood

> This is the flagship template. It proves SynApps can power 2Brain's capture pipeline.

### What It Does

The 2Brain Inbox Triage workflow takes any raw thought, note, or idea and automatically:

1. **Classifies** it into one of 4 categories: `idea`, `task`, `reference`, or `note`
2. **Structures** the classification into a JSON object with metadata
3. **Stores** the structured result in SynApps' memory system

This is the core "capture → classify → store" loop that 2Brain needs. Instead of you manually filing every thought, SynApps does the triage.

### Pipeline

```
Start (Raw Inbox Item)
  │
  ▼
Ollama Classifier (LLM)          ← llama3.1, temperature 0.1
  │  "Classify into: idea / task / reference / note"
  ▼
Structure Output (Code/Python)   ← Wraps result in JSON with timestamp + tags
  │
  ▼
Store in 2Brain (Memory)         ← namespace: "2brain", key: "2brain-inbox"
  │
  ▼
End (Triaged Item)
```

### Setup

1. **Ollama must be running** in WSL with `llama3.1` pulled:
   ```bash
   ollama serve
   ollama pull llama3.1
   ```
   Verify: `curl http://localhost:11434/api/tags` should list `llama3.1`.

2. **Create the workflow**: Dashboard → Create New Workflow → select "2Brain Inbox Triage" → Create Flow.

3. **Verify the pipeline**: You should see 5 nodes in a vertical line, all connected. Double-click "Ollama Classifier" to confirm: provider=ollama, model=llama3.1.

### Test Runs

Save the workflow, then try each of these inputs in the "Input Data" textarea:

| # | Input | Expected Category |
|---|-------|-------------------|
| 1 | `{"text": "Build a CLI tool for capturing voice notes"}` | `task` |
| 2 | `{"text": "What if we used vector search for memory retrieval?"}` | `idea` |
| 3 | `{"text": "RFC 9110 defines HTTP semantics"}` | `reference` |
| 4 | `{"text": "Had a good meeting with the team today"}` | `note` |
| 5 | `{"text": "Buy groceries"}` | `task` |

For each run:
- Click "Run Workflow" and wait for completion
- Check Output Data panel — you should see structured JSON like:
  ```json
  {
    "category": "task",
    "content": "Build a CLI tool for capturing voice notes",
    "captured_at": "2026-02-22T...",
    "tags": ["task"]
  }
  ```
- Was the classification correct? Note any misclassifications.
- How long did each run take? (Ollama local inference is typically 1-5s)

### What 2Brain Needs from SynApps

For 2Brain to use SynApps as its backend workflow engine:

1. **API-triggered runs** — 2Brain needs to POST to `/api/v1/flows/{id}/run` with input data and get results back. This already works.
2. **Memory persistence** — Items stored via the Memory node need to survive across runs. Currently memory is in-process (resets on backend restart). For production, this needs a persistent store.
3. **Batch capture** — 2Brain captures dozens of items daily. For-Each node can process arrays, but the template currently handles one item at a time. A batch variant would wrap this in a For-Each.
4. **Custom categories** — The current 4-way classification (idea/task/reference/note) is hardcoded in the LLM system prompt. Making this configurable (via node config or input data) would let 2Brain users define their own taxonomy.
5. **Confidence scoring** — The LLM returns a single word. Adding a confidence score (e.g., "task: 0.92") would let 2Brain flag uncertain classifications for human review.

### Rough Edges (Be Honest)

Things you may notice during the dogfood:

- **Start/End node labels**: The template sets `data.label` to "Raw Inbox Item" and "Triaged Item", but Start/End nodes always display "Start" and "End". The custom labels are stored but not rendered.
- **No image generator should appear**: The Stability AI dropdown only shows for workflows with Artist nodes. If it shows up for 2Brain, that's a bug (was fixed in `edb7065`).
- **Memory is ephemeral**: The "Store in 2Brain" node writes to in-memory storage. Restart the backend and it's gone. This is a known limitation, not a bug.
- **Ollama cold start**: First run after pulling the model may take 10-20s while Ollama loads weights into memory. Subsequent runs are faster.

---

## Verdict Template

After completing the walkthrough, fill this out:

### Overall Grade: _____ / F

### Top 3 Delights
1.
2.
3.

### Top 3 Friction Points
1.
2.
3.

### Would You Use This to Build a Real Workflow? (Yes / With Caveats / No)

Explain:

### One Thing to Fix Before Showing Anyone Else



---

## Known Limitations (Not Bugs)

These are documented and intentional for v1.0:

1. **Settings API keys** go to localStorage, not the backend. LLM nodes use server-side env vars for actual API calls.
2. **All users share all flows** — no per-user isolation yet.
3. **Code Editor "View Code"** shows placeholder boilerplate, not real applet source.
4. **`/ai/suggest` endpoint** returns 501 (not implemented).
5. **Docker nginx** doesn't proxy WebSocket — use dev mode for UAT.
6. **Memory is in-process** — restarts clear stored data.

---

## Bug Report Template

```
## Bug: [Short title]
**Page**: Dashboard / Editor / History / Settings / Applets
**Steps**: 1. ... 2. ... 3. ...
**Expected**: ...
**Actual**: ...
**Console errors**: (paste from browser DevTools > Console)
**Screenshot**: (if visual)
```
