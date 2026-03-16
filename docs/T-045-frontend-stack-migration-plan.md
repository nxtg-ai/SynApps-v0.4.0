# T-045: Frontend Stack Migration Plan

**Author:** Claude (Forge Architect)
**Date:** 2026-02-17
**Status:** Design Complete

---

## Executive Summary

This document plans the migration of the SynApps web frontend from:

| Layer | Current | Target |
|-------|---------|--------|
| **Build Tool** | Create React App (react-scripts 5.0.1) | Vite 6.x |
| **Styling** | Plain CSS (16 files, ~1,800 lines) | Tailwind CSS 4 + shadcn/ui |
| **State Management** | Component-local state + service singletons | Zustand stores |
| **TypeScript** | 4.8 | 5.x (Vite requirement) |
| **Node (Docker)** | 16-alpine | 20-alpine (LTS) |

The migration touches **every source file** but can be executed incrementally across 5 phases without a full rewrite.

---

## 1. Phase 1: CRA to Vite Migration

### 1.1 Why Vite

- **10-50x faster** dev server startup (ESM-native, no bundling in dev)
- **Instant HMR** vs CRA's full-page Webpack rebuilds
- CRA is officially deprecated/unmaintained
- Native ESM imports, first-class TypeScript support
- Smaller production bundles via Rollup

### 1.2 Steps

#### 1.2.1 Install Vite and plugins

```bash
cd apps/web-frontend
npm install --save-dev vite @vitejs/plugin-react
npm uninstall react-scripts
```

Remove `react-scripts` from `dependencies` in `package.json`.

#### 1.2.2 Create `vite.config.ts`

```ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
  build: {
    outDir: 'dist',       // Vite default (was 'build' in CRA)
    sourcemap: true,
  },
});
```

#### 1.2.3 Move and update `index.html`

- Move `public/index.html` to project root (`apps/web-frontend/index.html`)
- Remove `%PUBLIC_URL%` placeholders (Vite uses relative paths)
- Add the Vite entry script tag:

```html
<!-- Remove: no script tag in CRA's index.html -->
<!-- Add before </body>: -->
<script type="module" src="/src/index.tsx"></script>
```

#### 1.2.4 Update `tsconfig.json`

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "jsx": "react-jsx",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["src", "vite-env.d.ts"]
}
```

#### 1.2.5 Add Vite env type declaration

Create `src/vite-env.d.ts`:
```ts
/// <reference types="vite/client" />
```

#### 1.2.6 Migrate environment variables

| CRA Pattern | Vite Pattern |
|-------------|-------------|
| `process.env.REACT_APP_API_URL` | `import.meta.env.VITE_API_URL` |
| `process.env.REACT_APP_WEBSOCKET_URL` | `import.meta.env.VITE_WEBSOCKET_URL` |

**Files requiring env var updates:**
- `src/services/ApiService.ts` (line 18)
- `src/services/WebSocketService.ts` (line 30)

Create `.env` / `.env.development`:
```env
VITE_API_URL=http://localhost:8000
VITE_WEBSOCKET_URL=ws://localhost:8000/ws
```

#### 1.2.7 Update `package.json` scripts

```json
{
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview",
    "test": "vitest",
    "test:coverage": "vitest run --coverage"
  }
}
```

#### 1.2.8 Migrate tests from Jest to Vitest

```bash
npm install --save-dev vitest @testing-library/react @testing-library/jest-dom jsdom
npm uninstall @types/jest
```

Create `vitest.config.ts` (or merge into `vite.config.ts`):
```ts
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: './src/test-setup.ts',
  },
});
```

Create `src/test-setup.ts`:
```ts
import '@testing-library/jest-dom';
```

**Test files to update** (replace `jest.mock` with `vi.mock`):
- `src/components/WorkflowCanvas/nodes/AppletNode.test.tsx`
- `src/pages/HistoryPage/HistoryPage.test.tsx`

#### 1.2.9 Update Docker build

Update `infra/docker/Dockerfile.frontend`:
```dockerfile
FROM node:20-alpine as frontend-builder
WORKDIR /app
COPY apps/web-frontend/package.json apps/web-frontend/package-lock.json* ./
RUN npm ci
COPY apps/web-frontend/ .
RUN npm run build

FROM nginx:alpine
COPY --from=frontend-builder /app/dist /usr/share/nginx/html
COPY infra/docker/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

Key change: build output moves from `/app/build` to `/app/dist`.

#### 1.2.10 Clean up CRA artifacts

Delete these files/folders:
- `public/manifest.json` (if not needed for PWA)
- `src/react-app-env.d.ts` (if it exists)
- `config/` or any ejected CRA config folders

#### 1.2.11 Update path aliases in all imports

After adding the `@/` alias, update component imports from relative paths:

| Before | After |
|--------|-------|
| `import MainLayout from '../../components/Layout/MainLayout'` | `import MainLayout from '@/components/Layout/MainLayout'` |
| `import apiService from '../../services/ApiService'` | `import apiService from '@/services/ApiService'` |
| `import { Flow } from '../../types'` | `import { Flow } from '@/types'` |

This affects every file in `src/pages/` and some files in `src/components/`.

### 1.3 Files Modified in Phase 1

| File | Change |
|------|--------|
| `package.json` | Remove react-scripts, add vite/vitest |
| `tsconfig.json` | Update for Vite module resolution |
| `vite.config.ts` | **NEW** - Vite configuration |
| `src/vite-env.d.ts` | **NEW** - Vite type declarations |
| `index.html` | **MOVED** from public/ to root, add script tag |
| `src/services/ApiService.ts` | `process.env.REACT_APP_*` → `import.meta.env.VITE_*` |
| `src/services/WebSocketService.ts` | Same env var migration |
| `src/test-setup.ts` | **NEW** - Vitest setup |
| `*.test.tsx` (2 files) | `jest.*` → `vi.*` |
| `infra/docker/Dockerfile.frontend` | Node 20, `dist` output dir |
| All `src/pages/**/*.tsx` | Import path aliases |
| All `src/components/**/*.tsx` | Import path aliases (deep imports) |

---

## 2. Phase 2: Tailwind CSS 4 Integration

### 2.1 Why Tailwind 4

- **Zero-config** CSS engine (no `tailwind.config.js` needed in v4)
- Utility-first approach eliminates 16 separate CSS files
- Design tokens via CSS `@theme` directive
- Built-in container queries, `@starting-style`, color-mix
- Smaller CSS output via automatic dead-code elimination

### 2.2 Steps

#### 2.2.1 Install Tailwind CSS 4

```bash
npm install tailwindcss @tailwindcss/vite
```

#### 2.2.2 Add Tailwind Vite plugin

Update `vite.config.ts`:
```ts
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // ...
});
```

#### 2.2.3 Set up the main CSS entry point

Replace `src/index.css` content with:
```css
@import "tailwindcss";

@theme {
  --color-primary: #1890ff;
  --color-success: #52c41a;
  --color-error: #ff4d4f;
  --color-warning: #faad14;
  --color-text: #333333;
  --color-text-secondary: #666666;
  --color-text-light: #999999;
  --color-border: #e8e8e8;
  --color-bg-light: #f5f5f5;
  --color-bg-hover: #f0f0f0;
  --color-bg-active: #e6f7ff;
  --color-sidebar: #001529;
  --color-bg-main: #f0f2f5;

  --font-sans: Inter, system-ui, -apple-system, sans-serif;
  --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
}
```

This maps the existing CSS custom properties to Tailwind theme tokens.

#### 2.2.4 Migration strategy for CSS files (incremental)

**Do NOT delete all CSS at once.** Migrate component-by-component:

1. Keep existing CSS imports working during migration
2. Convert one component at a time to Tailwind classes
3. Delete the `.css` file only after its component is fully converted
4. Use `@apply` sparingly for complex/repeated patterns only

#### 2.2.5 CSS file conversion order (by complexity, ascending)

| Priority | CSS File | Lines | Complexity | Notes |
|----------|----------|-------|------------|-------|
| 1 | `NotFoundPage.css` | ~30 | Low | Simple centered layout |
| 2 | `App.css` | ~20 | Low | Minimal global styles |
| 3 | `TemplateLoader.css` | ~60 | Low | Modal + card grid |
| 4 | `MainLayout.css` | ~109 | Medium | Sidebar + header layout |
| 5 | `Notifications.css` | ~80 | Medium | Bell + dropdown panel |
| 6 | `DashboardPage.css` | ~120 | Medium | Card grid + welcome section |
| 7 | `CodeEditor.css` | ~80 | Medium | Monaco wrapper |
| 8 | `SettingsPage.css` | ~150 | Medium | Form inputs + sections |
| 9 | `AppletLibraryPage.css` | ~130 | Medium | Search + details view |
| 10 | `NodeContextMenu.css` | ~82 | Medium | Right-click menu |
| 11 | `NodeConfigModal.css` | ~220 | Medium-High | Config form modal |
| 12 | `HistoryPage.css` | ~200 | High | Sidebar + details + tree |
| 13 | `Nodes.css` | ~269 | High | 6 node types, status states |
| 14 | `WorkflowCanvas.css` | ~263 | High | ReactFlow overrides, animations |
| 15 | `EditorPage.css` | ~300 | High | Most complex layout |

**Total:** ~16 CSS files, ~1,800 lines to convert.

#### 2.2.6 ReactFlow styling considerations

ReactFlow ships its own CSS. Tailwind will NOT replace ReactFlow's internal styles. Keep a small `workflow-canvas.css` for:
- ReactFlow theme overrides (`.react-flow__node`, `.react-flow__edge`)
- Custom node styling that must target ReactFlow's DOM structure
- Animation keyframes used by anime.js integration

Use Tailwind for the surrounding layout and non-ReactFlow elements.

### 2.3 Files Modified in Phase 2

| File | Change |
|------|--------|
| `vite.config.ts` | Add `@tailwindcss/vite` plugin |
| `src/index.css` | Replace with Tailwind import + `@theme` |
| 16 `*.css` files | **DELETED** incrementally |
| 14 `*.tsx` component files | Add Tailwind classes, remove CSS imports |
| `src/components/WorkflowCanvas/workflow-canvas.css` | **NEW** - Minimal ReactFlow overrides |

---

## 3. Phase 3: shadcn/ui Component Library

### 3.1 Why shadcn/ui

- **Copy-paste components**, not a black-box library - full control over code
- Built on Radix UI primitives (accessible, keyboard-navigable)
- Tailwind-styled by default - integrates naturally with Phase 2
- Components are added to the project source, not `node_modules`
- Active maintenance, large ecosystem

### 3.2 Steps

#### 3.2.1 Initialize shadcn/ui

```bash
npx shadcn@latest init
```

Configuration choices:
- Style: **New York** (more polished, dense)
- Base color: **Blue** (matches existing `--primary-color: #1890ff`)
- CSS variables: **Yes**
- Path alias: `@/components` (already set in Phase 1)
- Components location: `src/components/ui/`

This creates:
- `components.json` - shadcn configuration
- `src/lib/utils.ts` - `cn()` helper (clsx + tailwind-merge)

#### 3.2.2 Install required shadcn components

Based on the current UI patterns used across the app:

```bash
# Layout & Navigation
npx shadcn@latest add sidebar
npx shadcn@latest add navigation-menu
npx shadcn@latest add breadcrumb

# Overlays & Modals
npx shadcn@latest add dialog
npx shadcn@latest add sheet
npx shadcn@latest add popover
npx shadcn@latest add dropdown-menu
npx shadcn@latest add context-menu
npx shadcn@latest add tooltip

# Forms & Input
npx shadcn@latest add button
npx shadcn@latest add input
npx shadcn@latest add textarea
npx shadcn@latest add select
npx shadcn@latest add switch
npx shadcn@latest add label
npx shadcn@latest add form

# Data Display
npx shadcn@latest add card
npx shadcn@latest add badge
npx shadcn@latest add table
npx shadcn@latest add tabs
npx shadcn@latest add separator
npx shadcn@latest add scroll-area
npx shadcn@latest add skeleton

# Feedback
npx shadcn@latest add alert
npx shadcn@latest add toast
npx shadcn@latest add sonner
npx shadcn@latest add progress
```

#### 3.2.3 Component replacement mapping

| Current Pattern | Replace With | Affected Files |
|----------------|-------------|----------------|
| Custom modal divs | `<Dialog>` | NodeConfigModal, TemplateLoader, CodeEditor |
| Custom context menu | `<ContextMenu>` | NodeContextMenu |
| Custom sidebar nav | `<Sidebar>` + `<NavigationMenu>` | MainLayout |
| `<button>` elements | `<Button>` | All pages |
| `<input>` elements | `<Input>` | EditorPage, SettingsPage |
| `<textarea>` elements | `<Textarea>` | EditorPage |
| `<select>` elements | `<Select>` | SettingsPage |
| Custom notification panel | `<Sonner>` (toasts) + `<Popover>` | NotificationCenter |
| Custom card layouts | `<Card>` | DashboardPage, AppletLibraryPage |
| Custom tabs | `<Tabs>` | SettingsPage |
| Loading spinners | `<Skeleton>` | DashboardPage, HistoryPage |
| Status badges | `<Badge>` | AppletNode, HistoryPage |
| Custom toggle switches | `<Switch>` | SettingsPage |

#### 3.2.4 Migration approach per component

**MainLayout** (high impact - do first):
1. Replace custom sidebar with shadcn `<Sidebar>` + `<SidebarMenu>`
2. Replace header with proper app shell layout
3. All pages inherit the new layout automatically

**NodeConfigModal** → `<Dialog>`:
1. Wrap content in `<Dialog>` / `<DialogContent>` / `<DialogHeader>`
2. Replace custom form inputs with `<Input>`, `<Select>`, `<Label>`

**NodeContextMenu** → `<ContextMenu>`:
1. Direct replacement - shadcn's ContextMenu matches the use case exactly
2. Benefits: keyboard navigation, proper focus management

**TemplateLoader** → `<Dialog>` + `<Card>`:
1. Template selection modal becomes `<Dialog>`
2. Template cards become `<Card>` components

**NotificationCenter** → `<Popover>` + `<Sonner>`:
1. Bell dropdown becomes `<Popover>`
2. Real-time notifications use `<Sonner>` toasts
3. Notification list uses `<ScrollArea>`

### 3.3 Files Modified in Phase 3

| File | Change |
|------|--------|
| `components.json` | **NEW** - shadcn configuration |
| `src/lib/utils.ts` | **NEW** - `cn()` utility |
| `src/components/ui/*.tsx` | **NEW** - ~20 shadcn components |
| `src/components/Layout/MainLayout.tsx` | Rewrite with shadcn Sidebar |
| `src/components/WorkflowCanvas/NodeContextMenu.tsx` | Replace with shadcn ContextMenu |
| `src/components/WorkflowCanvas/NodeConfigModal.tsx` | Replace with shadcn Dialog |
| `src/components/TemplateLoader/TemplateLoader.tsx` | Replace with shadcn Dialog + Card |
| `src/components/Notifications/NotificationCenter.tsx` | Replace with shadcn Popover + Sonner |
| `src/components/CodeEditor/CodeEditor.tsx` | Replace modal with shadcn Dialog |
| All 6 page components | Use shadcn Button, Input, Card, etc. |

---

## 4. Phase 4: Zustand State Management

### 4.1 Why Zustand

- **Minimal boilerplate** - no providers, reducers, or actions ceremony
- Works outside React (services can read/write stores directly)
- Built-in middleware: `persist`, `devtools`, `immer`
- TypeScript-first with full type inference
- Tiny bundle size (~1KB)
- Natural fit for the existing singleton service pattern

### 4.2 Store Design

Based on analysis of the current state scattered across components:

#### 4.2.1 `src/stores/flowStore.ts` - Workflow State

Consolidates state currently in EditorPage (760 lines), DashboardPage, and WorkflowCanvas.

```ts
interface FlowStore {
  // State
  flows: Flow[];
  currentFlow: Flow | null;
  isLoading: boolean;
  isSaving: boolean;
  isDirty: boolean;

  // Actions
  fetchFlows: () => Promise<void>;
  fetchFlow: (flowId: string) => Promise<void>;
  saveFlow: (flow: Flow) => Promise<string>;
  deleteFlow: (flowId: string) => Promise<void>;
  updateNodes: (nodes: FlowNode[]) => void;
  updateEdges: (edges: FlowEdge[]) => void;
  setCurrentFlow: (flow: Flow | null) => void;
  createFromTemplate: (template: FlowTemplate) => void;
}
```

#### 4.2.2 `src/stores/executionStore.ts` - Run State

Consolidates state currently in EditorPage, HistoryPage, and WorkflowCanvas.

```ts
interface ExecutionStore {
  // State
  runs: WorkflowRunStatus[];
  currentRun: WorkflowRunStatus | null;
  isRunning: boolean;

  // Actions
  fetchRuns: () => Promise<void>;
  fetchRun: (runId: string) => Promise<void>;
  startRun: (flowId: string, inputData: Record<string, any>) => Promise<string>;
  updateRunStatus: (status: WorkflowRunStatus) => void;
  setCurrentRun: (run: WorkflowRunStatus | null) => void;
}
```

#### 4.2.3 `src/stores/appletStore.ts` - Applet Registry

Consolidates state in AppletLibraryPage and EditorPage.

```ts
interface AppletStore {
  // State
  applets: AppletMetadata[];
  isLoading: boolean;

  // Actions
  fetchApplets: () => Promise<void>;
  getAppletByType: (type: string) => AppletMetadata | undefined;
}
```

#### 4.2.4 `src/stores/notificationStore.ts` - Notifications

Consolidates state currently in NotificationCenter.

```ts
interface NotificationStore {
  // State
  notifications: NotificationItem[];
  unreadCount: number;

  // Actions
  addNotification: (notification: Omit<NotificationItem, 'id' | 'timestamp' | 'read'>) => void;
  markAsRead: (id: string) => void;
  markAllAsRead: () => void;
  clearAll: () => void;
}
```

#### 4.2.5 `src/stores/settingsStore.ts` - User Settings

Consolidates localStorage usage in SettingsPage. Uses Zustand `persist` middleware.

```ts
interface SettingsStore {
  // State (persisted to localStorage)
  openaiKey: string;
  stabilityKey: string;
  theme: 'light' | 'dark' | 'system';
  notificationsEnabled: boolean;

  // Actions
  updateSetting: <K extends keyof SettingsStore>(key: K, value: SettingsStore[K]) => void;
}
```

### 4.3 WebSocket Integration

The key architectural decision: **WebSocketService writes directly to Zustand stores**.

```ts
// In WebSocketService.ts - connect stores to WebSocket events
import { useExecutionStore } from '@/stores/executionStore';
import { useNotificationStore } from '@/stores/notificationStore';

// Inside handleMessage:
private handleWorkflowStatus(status: WorkflowRunStatus): void {
  // Direct store update - no React required
  useExecutionStore.getState().updateRunStatus(status);
  useNotificationStore.getState().addNotification({
    title: status.status === 'success' ? 'Workflow completed' : 'Workflow failed',
    message: `Workflow ${status.flow_id} ${status.status}`,
    type: status.status === 'error' ? 'error' : 'success',
  });
}
```

This eliminates the current callback-subscription pattern in NotificationCenter and WorkflowCanvas.

### 4.4 Migration Strategy

Migrate stores one at a time. Each store migration follows this pattern:

1. Create the Zustand store file
2. Move API calls from component into store actions
3. Replace `useState`/`useEffect` data-fetching with `useXxxStore()` selectors
4. Remove the component-local state
5. Verify the component still works

**Order:**
1. `settingsStore` (simplest, self-contained, uses `persist`)
2. `notificationStore` (isolated, validates WebSocket integration)
3. `appletStore` (simple list, used by 2 components)
4. `flowStore` (core complexity, used by 4 components)
5. `executionStore` (depends on flowStore, WebSocket events)

### 4.5 Files Modified in Phase 4

| File | Change |
|------|--------|
| `package.json` | Add `zustand` dependency |
| `src/stores/flowStore.ts` | **NEW** |
| `src/stores/executionStore.ts` | **NEW** |
| `src/stores/appletStore.ts` | **NEW** |
| `src/stores/notificationStore.ts` | **NEW** |
| `src/stores/settingsStore.ts` | **NEW** |
| `src/stores/index.ts` | **NEW** - Re-exports |
| `src/services/WebSocketService.ts` | Connect to stores instead of callbacks |
| `src/services/ApiService.ts` | No change (stores call apiService) |
| `src/pages/EditorPage/EditorPage.tsx` | Replace ~15 useState with store selectors |
| `src/pages/DashboardPage/DashboardPage.tsx` | Replace state with store selectors |
| `src/pages/HistoryPage/HistoryPage.tsx` | Replace state with store selectors |
| `src/pages/AppletLibraryPage/AppletLibraryPage.tsx` | Replace state with store |
| `src/pages/SettingsPage/SettingsPage.tsx` | Replace localStorage with persist store |
| `src/components/Notifications/NotificationCenter.tsx` | Replace local state + WS callbacks |
| `src/components/WorkflowCanvas/WorkflowCanvas.tsx` | Replace run status callbacks |

---

## 5. Phase 5: Cleanup and Verification

### 5.1 Steps

1. **Remove deprecated dependencies**
   - `react-flow-renderer` (old, `reactflow` is already used)
   - `socket.io-client` (installed but unused)
   - Any remaining CRA dependencies

2. **Update path aliases** - Ensure all imports use `@/` prefix

3. **Verify Docker build** - Run `docker build` with new Vite output

4. **Run full test suite** - All tests pass under Vitest

5. **Performance audit** - Compare dev startup time and production bundle size

6. **Update CI/CD** - Ensure GitHub Actions uses updated build commands

---

## 6. Risk Assessment

### High Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| ReactFlow CSS breaks during Tailwind migration | Canvas becomes unusable | Keep a dedicated `workflow-canvas.css` for RF overrides; migrate ReactFlow-adjacent styles last |
| Monaco Editor conflicts with Vite | Code editor breaks | Monaco has a dedicated Vite plugin (`@monaco-editor/react` works with Vite); test early in Phase 1 |
| Anime.js animation timing changes | Visual glitches in workflow execution | Animations are CSS-independent (JS-driven); should be unaffected, but test manually |

### Medium Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| Zustand store hydration race with WebSocket | Missed real-time updates on page load | Initialize WebSocket connection after stores are ready; add message queue buffer |
| shadcn/ui theme clash with existing colors | Inconsistent visual design | Map existing CSS variables to shadcn's CSS variable system in Phase 3 setup |
| Test mocks break in Vitest migration | False test failures | `vitest` is Jest-compatible; only `jest.mock` → `vi.mock` syntax changes needed |
| Build output directory change (`build` → `dist`) | Broken deployment | Update Dockerfile, nginx config, CI scripts, and any deployment references |

### Low Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| Path alias `@/` breaks IDE autocomplete | Developer inconvenience | VS Code reads `tsconfig.json` paths; works automatically |
| `process.env` references missed during env var migration | Runtime errors | Global search for `process.env.REACT_APP_` to find all occurrences (only 2 files) |
| TypeScript version bump introduces new errors | Build failures | `strict: true` is already enabled; TS 5 is backward-compatible |

---

## 7. Dependency Changes Summary

### Added

| Package | Version | Purpose |
|---------|---------|---------|
| `vite` | ^6.x | Build tool |
| `@vitejs/plugin-react` | ^4.x | React Fast Refresh for Vite |
| `@tailwindcss/vite` | ^4.x | Tailwind CSS 4 Vite integration |
| `tailwindcss` | ^4.x | Utility-first CSS framework |
| `zustand` | ^5.x | State management |
| `vitest` | ^3.x | Test runner |
| `jsdom` | ^26.x | Test environment |
| `clsx` | ^2.x | Conditional class names (shadcn dep) |
| `tailwind-merge` | ^3.x | Merge Tailwind classes (shadcn dep) |
| ~20 Radix UI packages | various | shadcn/ui primitives |

### Removed

| Package | Reason |
|---------|--------|
| `react-scripts` | Replaced by Vite |
| `react-flow-renderer` | Deprecated; `reactflow` already present |
| `socket.io-client` | Unused (native WebSocket is used) |
| `@types/jest` | Replaced by Vitest globals |
| `webpack-dev-server` | No longer needed |

### Unchanged

| Package | Notes |
|---------|-------|
| `react`, `react-dom` | Stay at 18.x |
| `react-router-dom` | Stay at 6.x |
| `reactflow` | Stay at 11.x |
| `@monaco-editor/react` | Compatible with Vite |
| `axios` | Used by Zustand store actions |
| `animejs` | JS-driven, build-tool agnostic |

---

## 8. Complete File Impact Matrix

### New Files (14)

| File | Phase |
|------|-------|
| `vite.config.ts` | 1 |
| `src/vite-env.d.ts` | 1 |
| `src/test-setup.ts` | 1 |
| `.env` / `.env.development` | 1 |
| `components.json` | 3 |
| `src/lib/utils.ts` | 3 |
| `src/components/ui/*.tsx` (~20 files) | 3 |
| `src/stores/flowStore.ts` | 4 |
| `src/stores/executionStore.ts` | 4 |
| `src/stores/appletStore.ts` | 4 |
| `src/stores/notificationStore.ts` | 4 |
| `src/stores/settingsStore.ts` | 4 |
| `src/stores/index.ts` | 4 |
| `src/components/WorkflowCanvas/workflow-canvas.css` | 2 |

### Modified Files (24)

| File | Phases | Changes |
|------|--------|---------|
| `package.json` | 1, 2, 4, 5 | Deps, scripts |
| `tsconfig.json` | 1 | Module resolution, paths |
| `index.html` | 1 | Moved, script tag added |
| `src/index.tsx` | 1, 2 | CSS import update |
| `src/index.css` | 2 | Tailwind import + theme |
| `src/App.tsx` | 1, 2, 3 | Path aliases, Tailwind, Sonner provider |
| `src/services/ApiService.ts` | 1 | Env vars |
| `src/services/WebSocketService.ts` | 1, 4 | Env vars, store integration |
| `src/components/Layout/MainLayout.tsx` | 1, 2, 3 | shadcn Sidebar |
| `src/components/WorkflowCanvas/WorkflowCanvas.tsx` | 1, 2, 4 | Aliases, Tailwind, store |
| `src/components/WorkflowCanvas/NodeContextMenu.tsx` | 1, 2, 3 | shadcn ContextMenu |
| `src/components/WorkflowCanvas/NodeConfigModal.tsx` | 1, 2, 3 | shadcn Dialog |
| `src/components/WorkflowCanvas/nodes/AppletNode.tsx` | 1, 2, 3 | Tailwind, Badge |
| `src/components/WorkflowCanvas/nodes/StartNode.tsx` | 1, 2 | Tailwind |
| `src/components/WorkflowCanvas/nodes/EndNode.tsx` | 1, 2 | Tailwind |
| `src/components/CodeEditor/CodeEditor.tsx` | 1, 2, 3 | shadcn Dialog |
| `src/components/TemplateLoader/TemplateLoader.tsx` | 1, 2, 3 | shadcn Dialog + Card |
| `src/components/Notifications/NotificationCenter.tsx` | 1, 2, 3, 4 | Full rewrite |
| `src/pages/DashboardPage/DashboardPage.tsx` | 1, 2, 3, 4 | All phases |
| `src/pages/EditorPage/EditorPage.tsx` | 1, 2, 3, 4 | All phases (heaviest) |
| `src/pages/HistoryPage/HistoryPage.tsx` | 1, 2, 3, 4 | All phases |
| `src/pages/AppletLibraryPage/AppletLibraryPage.tsx` | 1, 2, 3, 4 | All phases |
| `src/pages/SettingsPage/SettingsPage.tsx` | 1, 2, 3, 4 | All phases |
| `src/pages/NotFoundPage/NotFoundPage.tsx` | 1, 2 | Aliases, Tailwind |
| `infra/docker/Dockerfile.frontend` | 1 | Node 20, dist dir |

### Deleted Files (18)

| File | Phase | Reason |
|------|-------|--------|
| `src/App.css` | 2 | Replaced by Tailwind |
| `src/components/Layout/MainLayout.css` | 2 | Replaced by Tailwind |
| `src/components/WorkflowCanvas/WorkflowCanvas.css` | 2 | Split: Tailwind + workflow-canvas.css |
| `src/components/WorkflowCanvas/NodeContextMenu.css` | 2 | Replaced by shadcn |
| `src/components/WorkflowCanvas/NodeConfigModal.css` | 2 | Replaced by shadcn |
| `src/components/WorkflowCanvas/nodes/Nodes.css` | 2 | Replaced by Tailwind |
| `src/components/CodeEditor/CodeEditor.css` | 2 | Replaced by Tailwind |
| `src/components/TemplateLoader/TemplateLoader.css` | 2 | Replaced by Tailwind |
| `src/components/Notifications/Notifications.css` | 2 | Replaced by shadcn |
| `src/pages/DashboardPage/DashboardPage.css` | 2 | Replaced by Tailwind |
| `src/pages/EditorPage/EditorPage.css` | 2 | Replaced by Tailwind |
| `src/pages/HistoryPage/HistoryPage.css` | 2 | Replaced by Tailwind |
| `src/pages/AppletLibraryPage/AppletLibraryPage.css` | 2 | Replaced by Tailwind |
| `src/pages/SettingsPage/SettingsPage.css` | 2 | Replaced by Tailwind |
| `src/pages/NotFoundPage/NotFoundPage.css` | 2 | Replaced by Tailwind |
| `public/manifest.json` | 1 | CRA artifact (unless PWA needed) |
| `src/react-app-env.d.ts` | 1 | CRA type file, replaced by vite-env.d.ts |

---

## 9. Implementation Timeline

| Phase | Description | Estimated Scope | Dependencies |
|-------|-------------|----------------|--------------|
| **Phase 1** | CRA → Vite | ~30 file touches | None |
| **Phase 2** | Tailwind CSS 4 | ~30 file touches | Phase 1 |
| **Phase 3** | shadcn/ui | ~25 file touches | Phase 2 |
| **Phase 4** | Zustand | ~15 file touches | Phase 1 |
| **Phase 5** | Cleanup | ~5 file touches | All phases |

**Phase 1 and Phase 4 can run in parallel** since Zustand doesn't depend on the styling stack. However, Phase 2 and Phase 3 are sequential (shadcn requires Tailwind).

Recommended execution order: **Phase 1 → Phase 4 → Phase 2 → Phase 3 → Phase 5**

This front-loads the build tool migration (highest risk) and state management (highest architectural value), then layers on the visual changes.

---

## 10. Validation Checklist

After each phase, verify:

- [ ] `npm run dev` starts without errors
- [ ] `npm run build` produces a production bundle
- [ ] `npm run test` passes all existing tests
- [ ] All 6 pages render correctly
- [ ] WorkflowCanvas drag/drop and connections work
- [ ] WebSocket real-time updates arrive
- [ ] Monaco code editor loads and highlights
- [ ] Docker build succeeds
- [ ] No console errors in browser DevTools
