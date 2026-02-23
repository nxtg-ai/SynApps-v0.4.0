# SynApps API Reference

Base URL: `http://localhost:8000/api/v1`

Interactive docs: [Swagger UI](/api/v1/docs) | [ReDoc](/api/v1/redoc) | [OpenAPI JSON](/api/v1/openapi.json)

---

## Authentication

All endpoints (except health and auth) require a JWT bearer token or API key.

### Register

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "securepass123"}'
```

**Response** `201`:
```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer"
}
```

### Login

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "securepass123"}'
```

**Response** `200`: Same shape as register.

### Refresh Token

```bash
curl -X POST http://localhost:8000/api/v1/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "eyJ..."}'
```

### Logout

```bash
curl -X POST http://localhost:8000/api/v1/auth/logout \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "eyJ..."}'
```

### Get Current User

```bash
curl http://localhost:8000/api/v1/auth/me \
  -H "Authorization: Bearer <access_token>"
```

**Response** `200`:
```json
{
  "id": "uuid",
  "email": "user@example.com",
  "is_active": true,
  "created_at": "2026-02-22T00:00:00"
}
```

### API Keys

Create:
```bash
curl -X POST http://localhost:8000/api/v1/auth/api-keys \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-key"}'
```

List:
```bash
curl http://localhost:8000/api/v1/auth/api-keys \
  -H "Authorization: Bearer <access_token>"
```

Revoke:
```bash
curl -X DELETE http://localhost:8000/api/v1/auth/api-keys/<key_id> \
  -H "Authorization: Bearer <access_token>"
```

Use an API key via header:
```bash
curl http://localhost:8000/api/v1/flows \
  -H "X-API-Key: synapps_..."
```

---

## Flows (Workflows)

### Create Flow

```bash
curl -X POST http://localhost:8000/api/v1/flows \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Workflow",
    "nodes": [
      {"id": "start", "type": "start", "position": {"x": 300, "y": 25}, "data": {"label": "Start"}},
      {"id": "llm1", "type": "llm", "position": {"x": 300, "y": 150}, "data": {"label": "GPT-4o", "provider": "openai", "model": "gpt-4o"}},
      {"id": "end", "type": "end", "position": {"x": 300, "y": 300}, "data": {"label": "End"}}
    ],
    "edges": [
      {"id": "e1", "source": "start", "target": "llm1"},
      {"id": "e2", "source": "llm1", "target": "end"}
    ]
  }'
```

**Response** `201`:
```json
{"id": "uuid", "name": "My Workflow", "nodes": [...], "edges": [...]}
```

### List Flows

```bash
curl http://localhost:8000/api/v1/flows?page=1&page_size=20 \
  -H "Authorization: Bearer <token>"
```

**Response** `200`:
```json
{"items": [...], "total": 5, "page": 1, "page_size": 20, "total_pages": 1}
```

### Get Flow

```bash
curl http://localhost:8000/api/v1/flows/<flow_id> \
  -H "Authorization: Bearer <token>"
```

### Delete Flow

```bash
curl -X DELETE http://localhost:8000/api/v1/flows/<flow_id> \
  -H "Authorization: Bearer <token>"
```

### Export Flow

```bash
curl http://localhost:8000/api/v1/flows/<flow_id>/export \
  -H "Authorization: Bearer <token>" \
  -o workflow.synapps.json
```

### Import Flow

```bash
curl -X POST http://localhost:8000/api/v1/flows/import \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d @workflow.synapps.json
```

**Response** `201`:
```json
{"message": "Flow imported", "id": "new-uuid"}
```

---

## Runs (Workflow Execution)

### Execute a Flow

```bash
curl -X POST http://localhost:8000/api/v1/flows/<flow_id>/runs \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"input": {"text": "Hello, classify this thought"}}'
```

**Response** `202`:
```json
{"run_id": "uuid"}
```

### List Runs

```bash
curl http://localhost:8000/api/v1/runs?page=1&page_size=20 \
  -H "Authorization: Bearer <token>"
```

### Get Run

```bash
curl http://localhost:8000/api/v1/runs/<run_id> \
  -H "Authorization: Bearer <token>"
```

### Get Run Trace

Full execution trace with per-node timings and outputs:

```bash
curl http://localhost:8000/api/v1/runs/<run_id>/trace \
  -H "Authorization: Bearer <token>"
```

### Diff Two Runs

```bash
curl "http://localhost:8000/api/v1/runs/<run_id>/diff?other_run_id=<other_id>" \
  -H "Authorization: Bearer <token>"
```

### Re-run a Workflow

```bash
curl -X POST http://localhost:8000/api/v1/runs/<run_id>/rerun \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"input": {"text": "new input"}, "merge_with_original_input": false}'
```

---

## Providers

### LLM Providers

```bash
curl http://localhost:8000/api/v1/llm/providers \
  -H "Authorization: Bearer <token>"
```

**Response** `200`:
```json
{
  "items": [
    {
      "name": "openai",
      "configured": true,
      "reason": "",
      "models": [
        {"id": "gpt-4o", "name": "GPT-4o", "context_window": 128000, "supports_vision": true}
      ]
    }
  ],
  "total": 5, "page": 1, "page_size": 20, "total_pages": 1
}
```

Built-in providers: `openai`, `anthropic`, `google`, `ollama`, `custom`.

### Image Providers

```bash
curl http://localhost:8000/api/v1/image/providers \
  -H "Authorization: Bearer <token>"
```

Built-in providers: `openai`, `stability`, `flux`.

---

## Applets (Node Types)

```bash
curl http://localhost:8000/api/v1/applets \
  -H "Authorization: Bearer <token>"
```

Returns metadata for all registered node types: `start`, `end`, `llm`, `code`, `http_request`, `transform`, `if_else`, `merge`, `for_each`, `memory`, `image_gen`.

---

## Dashboard

### Portfolio Dashboard

Template statuses, provider registry, and health for internal portfolio consumers.

```bash
curl http://localhost:8000/api/v1/dashboard/portfolio \
  -H "Authorization: Bearer <token>"
```

**Response** `200`:
```json
{
  "templates": [
    {
      "id": "content-engine-pipeline",
      "name": "Content Engine Pipeline",
      "description": "Fetch content from a web source...",
      "tags": ["content-engine", "research", "portfolio"],
      "source": "templates/content_engine.yaml",
      "node_count": 6,
      "edge_count": 5,
      "last_run": null
    }
  ],
  "template_count": 1,
  "providers": [
    {"name": "openai", "configured": false, "reason": "OPENAI_API_KEY not set", "model_count": 4}
  ],
  "provider_count": 5,
  "health": {
    "status": "healthy",
    "database": "reachable",
    "uptime_seconds": 3600,
    "version": "1.0.0"
  }
}
```

---

## Health

```bash
curl http://localhost:8000/api/v1/health
```

**Response** `200`:
```json
{"status": "healthy", "service": "SynApps Orchestrator API", "version": "1.0.0", "uptime": 3600}
```

Also available at `/` and `/health` (unversioned).

---

## Node Types Reference

| Type | Description | Key Config Fields |
|------|-------------|-------------------|
| `start` | Entry point | `label` |
| `end` | Terminal node | `label` |
| `llm` | LLM completion | `provider`, `model`, `system_prompt`, `temperature`, `max_tokens` |
| `code` | Sandboxed Python/JS | `language`, `code`, `timeout_seconds`, `memory_limit_mb` |
| `http_request` | HTTP call | `method`, `url`, `headers`, `body` |
| `transform` | Text manipulation | `operation` (split, join, regex, template, json_path) |
| `if_else` | Conditional routing | `operation` (equals, contains, regex, json_path), `source`, `value` |
| `merge` | Fan-in combiner | `strategy` (array, concatenate, first_wins) |
| `for_each` | Loop iteration | `array_source`, `parallel`, `concurrency_limit` |
| `memory` | Key-value store | `operation` (store, retrieve, search, clear), `namespace` |
| `image_gen` | Image generation | `provider`, `model`, `prompt`, `size` |

---

## Pagination

All list endpoints return paginated responses:

```json
{"items": [...], "total": 100, "page": 1, "page_size": 20, "total_pages": 5}
```

Query params: `page` (default 1), `page_size` (default 20, max 100).

---

## Error Format

All errors follow a consistent structure:

```json
{
  "error": {
    "code": "NOT_FOUND",
    "message": "Flow not found",
    "details": {}
  }
}
```

Common codes: `VALIDATION_ERROR`, `NOT_FOUND`, `UNAUTHORIZED`, `RATE_LIMIT_EXCEEDED`, `INTERNAL_ERROR`.
