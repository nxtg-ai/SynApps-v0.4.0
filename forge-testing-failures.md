# Forge Testing Failures Report: Frontend Auth Wiring

**Date:** 2026-02-19
**Scope:** Frontend authentication implementation (login, register, token management, protected routes, auto-refresh)
**Codebase:** `apps/web-frontend/`

---

## Executive Summary

The dev team implemented 7 new files and modified 5 existing files to wire up frontend authentication. While the runtime vitest suite ends green (101/101), the implementation shipped with **11 pre-existing TypeScript type errors in test files**, **zero test coverage on all 7 new files**, **broken E2E tests** (auth redirects now intercept every E2E scenario), **unused imports carried forward**, and a **test mock that broke and had to be patched retroactively**. These issues fall into five failure categories documented below.

---

## Failure 1: ApiService Test Mock Did Not Match New Production Code

**Severity:** HIGH (10 tests failed immediately after implementation)
**Files:** `src/services/ApiService.ts`, `src/services/ApiService.test.ts`

### What happened

The auth implementation added a `request` interceptor to ApiService:

```ts
this.api.interceptors.request.use((config: InternalAxiosRequestConfig) => { ... });
```

The existing test file mocked `axios.create()` to return an instance with only `interceptors.response.use` — no `interceptors.request` existed on the mock. Every test that instantiated `ApiService` threw:

```
TypeError: Cannot read properties of undefined (reading 'use')
  at new ApiService src/services/ApiService.ts:43:35
```

All 10 `ApiService` tests failed.

### Root cause

The dev team modified production code that touched `this.api.interceptors` without checking whether the test mock's shape covered the new code path. The mock was a hand-rolled partial that only included `{ interceptors: { response: { use } } }` — it did not track the real axios instance shape.

### Fix applied

Added `interceptors.request.use` to the mock object and a corresponding assertion:

```diff
+  const requestUse = vi.fn();
   const instance = {
     interceptors: {
+      request: { use: requestUse },
       response: { use },
     },
   };
```

### Lesson for Forge models

**When modifying code that interacts with a mocked dependency, always verify the mock's shape covers all new call sites.** Before committing changes to an interceptor, constructor, or initialization path, read the corresponding test file and check that every property chain accessed in the new code also exists on the mock. A simple `grep interceptors src/services/ApiService.test.ts` would have caught this.

---

## Failure 2: Zero Test Coverage on All 7 New Files

**Severity:** HIGH (entire auth layer untested)
**Files affected:**

| New file | Lines | Test file | Tests |
|----------|-------|-----------|-------|
| `src/services/AuthService.ts` | 53 | none | 0 |
| `src/stores/authStore.ts` | 65 | none | 0 |
| `src/pages/LoginPage/LoginPage.tsx` | 85 | none | 0 |
| `src/pages/RegisterPage/RegisterPage.tsx` | 97 | none | 0 |
| `src/components/ProtectedRoute.tsx` | 18 | none | 0 |
| `src/components/GuestRoute.tsx` | 16 | none | 0 |
| `src/pages/LoginPage/LoginPage.css` | 108 | n/a | n/a |

### What happened

The team created an `AuthService` with 5 methods (register, login, refresh, logout, getMe), a Zustand `authStore` with 3 actions (loadAuth, setAuth, clearAuth), two full page components with form logic and error handling, and two route guards — all with **zero** unit or integration tests.

### What should have been tested

1. **AuthService:** Each method hits the correct endpoint with correct payload. Refresh sends `{ refresh_token }`. Error responses are propagated.
2. **authStore:** `loadAuth()` hydrates from localStorage. `setAuth()` persists tokens and user. `clearAuth()` removes all keys. Corrupt `auth_user` JSON falls through gracefully.
3. **ProtectedRoute:** Renders children when `isAuthenticated=true`. Redirects to `/login` when `false`. Returns `null` during `isLoading`.
4. **GuestRoute:** Renders children when `isAuthenticated=false`. Redirects to `/dashboard` when `true`.
5. **LoginPage:** Renders form fields. Calls `authService.login` on submit. Shows error on 401. Navigates to `/dashboard` on success.
6. **RegisterPage:** Validates password length (min 8). Validates passwords match. Calls `authService.register`. Shows backend errors.

### Lesson for Forge models

**Every new file containing logic must ship with a corresponding test file.** The project already had a clear convention: `ApiService.ts` has `ApiService.test.ts`, `settingsStore.ts` has `settingsStore.test.ts`. Following the existing pattern was the obvious move. The "tests pass" claim is hollow when 7 new files contributed 0 test cases to the suite.

---

## Failure 3: E2E Tests Broken by Auth Redirects

**Severity:** HIGH (all 4 happy-path E2E scenarios will fail at runtime)
**Files:** `e2e/happy-path.e2e.ts`, `e2e/example.e2e.ts`

### What happened

The E2E tests navigate directly to pages like `/dashboard`, `/editor/flow-created`, and `/settings`:

```ts
await page.goto('/');
await expect(page).toHaveURL(/\/dashboard$/);
```

After the auth changes, `ProtectedRoute` wraps every authenticated route. On page load, `loadAuth()` reads localStorage (empty in a fresh Playwright context), sets `isAuthenticated=false`, and `ProtectedRoute` immediately redirects to `/login`. Every E2E test that expects to land on `/dashboard` will instead land on `/login`.

The E2E test fixtures mock API responses (`page.route('**/flows', ...)`) but do **not**:
- Seed `localStorage` with `access_token` / `refresh_token` / `auth_user`
- Mock the `/auth/login` or `/auth/register` endpoints
- Navigate through the login flow first

### What should have been done

Either:
1. Add a `test.beforeEach` that seeds `localStorage` with valid auth tokens via `page.evaluate()`, or
2. Add a shared Playwright fixture/helper that performs the login flow before each test, or
3. Mock the auth endpoints and add a login step at the start of each E2E scenario.

### Lesson for Forge models

**When adding authentication guards to routes, check existing E2E tests for direct navigation.** Auth changes are a cross-cutting concern — they affect every page. The dev should have run `npx playwright test` (or at minimum searched E2E files for `page.goto`) to verify compatibility. Any auth implementation PR must include E2E fixture updates.

---

## Failure 4: Pre-existing TypeScript Errors in Test Files Ignored

**Severity:** MEDIUM (11 type errors across 3 test files)
**Files:**

| File | Errors | Nature |
|------|--------|--------|
| `src/utils/flowUtils.test.ts` | 8 | `sourceHandle`/`targetHandle` not in `FlowEdge` type |
| `src/stores/executionStore.test.ts` | 2 | `string` passed where `WorkflowRunStatus` expected |
| `src/services/WebSocketService.test.ts` | 1 | Implicit `any` on destructured `eventName` |

### What happened

`npx tsc --noEmit` reports 11 errors, all in test files. These are **not** caused by the auth changes — they were pre-existing. But the dev team:

1. Ran `tsc --noEmit`, saw errors, and declared "no auth-related type errors" — filtering by keyword rather than fixing.
2. Did not mention these errors in any PR notes or remediation plan.
3. Left the codebase in a state where `npm run typecheck` fails, meaning **the pre-commit hook for typecheck either doesn't run or is being bypassed.**

### Root causes

- **`flowUtils.test.ts`**: The `FlowEdge` interface in `src/types/index.ts` doesn't include `sourceHandle` or `targetHandle`, but the test fixtures use them. The runtime flow objects (from @xyflow/react) have these properties, but the local type definition is incomplete.
- **`executionStore.test.ts`**: The `runStatus` field is typed as `WorkflowRunStatus | null` (an object with `run_id`, `flow_id`, etc.), but the tests pass bare strings like `'running'` and `'completed'`. This indicates the store type was updated from a string enum to the full status object, but the tests weren't updated to match.
- **`WebSocketService.test.ts`**: A destructured callback parameter lacks a type annotation and `noImplicitAny` flags it.

### Lesson for Forge models

**Never ship code on top of a broken typecheck.** If `tsc --noEmit` fails, either fix the errors or explicitly document them as known tech debt with ticket references. Filtering errors by keyword ("no auth-related errors") and moving on normalizes a broken type system.

---

## Failure 5: Unused Imports Carried Forward

**Severity:** LOW (lint warnings, no runtime impact)
**Files:** `src/services/ApiService.ts`

### What happened

The original `ApiService.ts` had two unused imports: `AxiosRequestConfig` and `FlowTemplate`. The auth implementation added a new import (`InternalAxiosRequestConfig`) but left the dead imports in place. The result is 2 ESLint warnings on the file.

```ts
import axios, { AxiosInstance, AxiosRequestConfig, InternalAxiosRequestConfig } from 'axios';
import { Flow, FlowTemplate, AppletMetadata, ... } from '../types';
```

`AxiosRequestConfig` and `FlowTemplate` are never used anywhere in the file.

### Lesson for Forge models

**When modifying a file's import block, clean up existing dead imports.** It takes 5 seconds and prevents warning accumulation. The team was already editing line 4 — removing `AxiosRequestConfig` was free.

---

## Failure 6: No Auth-Aware WebSocket Reconnection

**Severity:** MEDIUM (silent failure on token expiry during long sessions)
**Files:** `src/services/WebSocketService.ts`

### What happened

The auth token is sent once on `socket.onopen`:

```ts
this.socket.onopen = () => {
  const token = window.localStorage.getItem('access_token');
  if (token && this.socket) {
    this.socket.send(JSON.stringify({ type: 'auth', token }));
  }
};
```

But when the WebSocket reconnects (after a disconnect/reconnect cycle via `scheduleReconnect`), it reads the token from localStorage again. If the access token has been refreshed by the API interceptor since the original connect, this works. But if the token has **expired** and the refresh hasn't happened yet (the HTTP interceptor only refreshes on 401 responses), the WebSocket will send a stale token.

There is also no handling for the case where the server rejects the auth message — no `auth_error` message handler, no re-auth flow.

### What should have been done

- Subscribe to authStore changes and reconnect when tokens change
- Handle a server-side `auth_error` WebSocket message to trigger a refresh
- Or: pass the token as a URL query parameter so the server validates on handshake

### Lesson for Forge models

**Auth on persistent connections (WebSocket, SSE) requires lifecycle management, not just send-once-on-open.** Token refresh, reconnection with fresh tokens, and server-side rejection handling are all required for production auth.

---

## Summary of Action Items

| # | Priority | Action | Files |
|---|----------|--------|-------|
| 1 | P0 | Write unit tests for all 7 new auth files | `AuthService.test.ts`, `authStore.test.ts`, `LoginPage.test.tsx`, `RegisterPage.test.tsx`, `ProtectedRoute.test.tsx`, `GuestRoute.test.tsx` |
| 2 | P0 | Fix E2E tests to seed auth state or mock auth endpoints | `e2e/happy-path.e2e.ts`, `e2e/example.e2e.ts` |
| 3 | P1 | Fix 11 pre-existing TypeScript errors in test files | `flowUtils.test.ts`, `executionStore.test.ts`, `WebSocketService.test.ts` |
| 4 | P1 | Add WebSocket auth lifecycle management | `WebSocketService.ts` |
| 5 | P2 | Remove unused imports from ApiService | `ApiService.ts` |
| 6 | P2 | Ensure pre-commit typecheck hook is enforced (not bypassable) | `.husky/`, `pre-commit` config |

---

## Forge Model Training Directives

These are the patterns to encode as rules:

1. **MOCK_SHAPE_SYNC**: When production code adds a new property access on a mocked dependency (e.g., `this.api.interceptors.request.use`), the test mock MUST be updated in the same commit. Verify by searching the test file for the mock object shape.

2. **NEW_FILE_NEW_TEST**: Every new `.ts`/`.tsx` file containing logic (not CSS/assets) MUST have a corresponding `.test.ts`/`.test.tsx` file in the same commit. No exceptions.

3. **AUTH_E2E_GUARD**: When adding route guards (auth, permissions, feature flags), search all E2E test files for `page.goto` calls. Any direct navigation to a guarded route needs auth seeding.

4. **TYPECHECK_ZERO_TOLERANCE**: `tsc --noEmit` must exit 0. Do not filter errors by keyword. If there are pre-existing errors, fix them or create a tracked issue — do not add more code on top.

5. **CLEAN_IMPORTS_ON_TOUCH**: When editing a file's import block, remove any unused imports in that block. Cost: 5 seconds. Benefit: zero lint warnings.

6. **PERSISTENT_CONNECTION_AUTH**: Auth on WebSocket/SSE connections requires: (a) send token on connect, (b) handle server-side auth rejection, (c) reconnect with fresh token after refresh. Send-once-on-open is insufficient.
