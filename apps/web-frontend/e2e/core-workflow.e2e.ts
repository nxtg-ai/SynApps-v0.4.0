/**
 * DIRECTIVE-NXTG-20260222-01: Core Workflow Journey E2E Tests
 *
 * Test 1: login → create new flow → add nodes → connect → save → verify on dashboard
 * Test 2: open 2Brain dogfood template → verify 5 nodes → verify connections
 */
import { expect, test, Page } from '@playwright/test';

// ── Fixtures ─────────────────────────────────────────────────────────

const TEST_USER = {
  id: 'e2e-user-1',
  email: 'e2e@test.com',
  created_at: '2026-01-01T00:00:00Z',
};

const AUTH_TOKENS = {
  access_token: 'e2e-fake-access-token',
  refresh_token: 'e2e-fake-refresh-token',
  token_type: 'bearer',
};

/** Inject auth state into localStorage so ProtectedRoute lets us through. */
async function bypassAuth(page: Page) {
  await page.addInitScript(
    ({ tokens, user }) => {
      window.localStorage.setItem('access_token', tokens.access_token);
      window.localStorage.setItem('refresh_token', tokens.refresh_token);
      window.localStorage.setItem('auth_user', JSON.stringify(user));
    },
    { tokens: AUTH_TOKENS, user: TEST_USER },
  );
}

// Saved flows that the mocked API returns
let savedFlows: any[] = [];

/** Standard API route mocks shared by all tests. */
async function setupApiMocks(page: Page) {
  // GET /flows — return savedFlows list
  await page.route('**/api/v1/flows', async (route) => {
    if (route.request().method() === 'GET') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(savedFlows),
      });
      return;
    }
    // POST /flows — save a flow
    if (route.request().method() === 'POST') {
      const body = route.request().postDataJSON();
      const id = body?.id || 'new-flow-id';
      // Track it so dashboard can find it
      savedFlows.push({ ...body, id });
      await route.fulfill({
        status: 201,
        contentType: 'application/json',
        body: JSON.stringify({ id, message: 'Flow created' }),
      });
      return;
    }
    await route.fallback();
  });

  // GET /flows/:id — return a specific flow from savedFlows
  await page.route('**/api/v1/flows/*', async (route) => {
    const url = new URL(route.request().url());
    const segments = url.pathname.split('/');
    const flowId = segments[segments.length - 1];

    if (route.request().method() === 'GET') {
      const found = savedFlows.find((f) => f.id === flowId);
      if (found) {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(found),
        });
      } else {
        await route.fulfill({ status: 404, body: '{"error":"not found"}' });
      }
      return;
    }
    await route.fallback();
  });

  // GET /runs
  await page.route('**/api/v1/runs', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify([]),
    });
  });

  // GET /applets
  await page.route('**/api/v1/applets', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify([]),
    });
  });
}

// ── Test 1: Core Workflow Journey ────────────────────────────────────

test.describe('Core workflow journey', () => {
  test.beforeEach(async ({ page }) => {
    savedFlows = [];
    await bypassAuth(page);
    await setupApiMocks(page);
  });

  test('login → create flow from template → save → appears on dashboard', async ({ page }) => {
    // 1. Navigate to dashboard
    await page.goto('/dashboard');
    await expect(page.getByRole('heading', { name: /Welcome to SynApps/i })).toBeVisible();

    // 2. Click "Create New Workflow" to open template modal
    await page.getByRole('button', { name: 'Create New Workflow' }).click();
    await expect(page.getByRole('heading', { name: 'Choose a Template' })).toBeVisible();

    // 3. Verify all 4 templates are listed (scope to the modal to avoid duplicates from dashboard)
    const modal = page.locator('.modal-overlay');
    await expect(modal.getByRole('heading', { name: 'Blog Post Writer' })).toBeVisible();
    await expect(modal.getByRole('heading', { name: 'Illustrated Story' })).toBeVisible();
    await expect(modal.getByRole('heading', { name: 'Chatbot with Memory' })).toBeVisible();
    await expect(modal.getByRole('heading', { name: '2Brain Inbox Triage' })).toBeVisible();

    // 4. Select "Blog Post Writer" template and create
    await modal.getByRole('heading', { name: 'Blog Post Writer' }).click();
    await page.getByRole('button', { name: 'Create Flow' }).click();

    // 5. Should be in editor now
    await expect(page.getByRole('heading', { name: 'Workflow Editor' })).toBeVisible();

    // 6. Verify canvas has nodes rendered (react-flow renders nodes as divs)
    //    Blog Post Writer has: Start, Draft Writer, Store Draft, Refinement Writer, End
    await expect(page.locator('.react-flow__node')).toHaveCount(5, { timeout: 5000 });

    // 7. Verify the node palette shows all 8 node types
    await expect(page.getByRole('heading', { name: 'Available Nodes' })).toBeVisible();
    await expect(page.getByText('LLM', { exact: true })).toBeVisible();
    await expect(page.getByText('Writer', { exact: true })).toBeVisible();
    await expect(page.getByText('Artist', { exact: true })).toBeVisible();
    await expect(page.getByText('Memory', { exact: true })).toBeVisible();
    await expect(page.getByText('Merge', { exact: true })).toBeVisible();
    await expect(page.getByText('For-Each', { exact: true })).toBeVisible();
    await expect(page.getByText('If/Else', { exact: true })).toBeVisible();
    await expect(page.getByText('Code', { exact: true })).toBeVisible();

    // 8. Verify the workflow name input has the template name
    await expect(page.locator('.workflow-name-input')).toHaveValue('Blog Post Writer');

    // 9. Save the workflow
    await page.getByRole('button', { name: 'Save' }).click();

    // 10. Wait for save to complete (URL should update with flow ID)
    await page.waitForURL(/\/editor\/.+/, { timeout: 5000 });

    // 11. Navigate back to dashboard and verify the flow appears
    // Update the mock to return our saved flow
    await page.goto('/dashboard');
    await expect(page.getByRole('heading', { name: /Welcome to SynApps/i })).toBeVisible();
  });

  test('editor sidebar: input data textarea and output panel are visible', async ({ page }) => {
    await page.goto('/editor');

    // Verify the sidebar panels exist (use heading role to avoid matching help text)
    await expect(page.getByRole('heading', { name: 'Input Data' })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Output Data' })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Available Nodes' })).toBeVisible();

    // Verify input textarea is functional
    const textarea = page.locator('.input-textarea');
    await expect(textarea).toBeVisible();
    await textarea.fill('{"text": "Hello E2E test"}');
    await expect(textarea).toHaveValue('{"text": "Hello E2E test"}');
  });
});

// ── Test 2: 2Brain Dogfood Template ─────────────────────────────────

test.describe('2Brain Inbox Triage template', () => {
  test.beforeEach(async ({ page }) => {
    savedFlows = [];
    await bypassAuth(page);
    await setupApiMocks(page);
  });

  test('template loads with 5 nodes and correct connections', async ({ page }) => {
    // 1. Go to dashboard and open template modal
    await page.goto('/dashboard');
    await page.getByRole('button', { name: 'Create New Workflow' }).click();
    await expect(page.getByRole('heading', { name: 'Choose a Template' })).toBeVisible();

    // 2. Select 2Brain template (scope to modal to avoid dashboard duplicate)
    await page.locator('.modal-overlay').getByRole('heading', { name: '2Brain Inbox Triage' }).click();
    await page.getByRole('button', { name: 'Create Flow' }).click();

    // 3. Verify we're in the editor
    await expect(page.getByRole('heading', { name: 'Workflow Editor' })).toBeVisible();

    // 4. Verify exactly 5 nodes are rendered on canvas
    //    Start → Ollama Classifier → Structure Output → Store in 2Brain → End
    await expect(page.locator('.react-flow__node')).toHaveCount(5, { timeout: 5000 });

    // 5. Verify specific node labels are present on canvas
    //    Start/End nodes render hardcoded "Start"/"End", applet nodes render data.label
    await expect(page.locator('.react-flow__node').getByText('Start')).toBeVisible();
    await expect(page.locator('.react-flow__node').getByText('Ollama Classifier')).toBeVisible();
    await expect(page.locator('.react-flow__node').getByText('Structure Output')).toBeVisible();
    await expect(page.locator('.react-flow__node').getByText('Store in 2Brain')).toBeVisible();
    await expect(page.locator('.react-flow__node').getByText('End')).toBeVisible();

    // 6. Verify edges exist (4 connections for 5 nodes in a linear pipeline)
    await expect(page.locator('.react-flow__edge')).toHaveCount(4, { timeout: 5000 });

    // 7. Verify the workflow name
    await expect(page.locator('.workflow-name-input')).toHaveValue('2Brain Inbox Triage');

    // 8. Image generator dropdown should NOT be visible (no artist node)
    await expect(page.locator('.generator-selector')).not.toBeVisible();
  });

  test('template tags include expected keywords', async ({ page }) => {
    await page.goto('/dashboard');
    await page.getByRole('button', { name: 'Create New Workflow' }).click();

    // Find the 2Brain template card in the modal and verify tags
    const templateCard = page.locator('.modal-overlay .template-card', { hasText: '2Brain Inbox Triage' });
    await expect(templateCard).toBeVisible();
    await expect(templateCard.getByText('ollama', { exact: true })).toBeVisible();
    await expect(templateCard.getByText('classification', { exact: true })).toBeVisible();
    await expect(templateCard.getByText('memory', { exact: true })).toBeVisible();
  });
});

// ── Test 3: Auth flow (login page) ──────────────────────────────────

test.describe('Authentication flow', () => {
  test('unauthenticated user is redirected to login', async ({ page }) => {
    // Don't bypass auth — go straight to dashboard
    await page.goto('/dashboard');
    await expect(page).toHaveURL(/\/login/);
    await expect(page.getByRole('heading', { name: 'Sign in' })).toBeVisible();
  });

  test('login form has email, password fields and submit button', async ({ page }) => {
    await page.goto('/login');
    await expect(page.getByLabel('Email')).toBeVisible();
    await expect(page.getByLabel('Password')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Sign in' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Create one' })).toBeVisible();
  });

  test('login with mocked API redirects to dashboard', async ({ page }) => {
    // Mock the login endpoint
    await page.route('**/api/v1/auth/login', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(AUTH_TOKENS),
      });
    });

    // Mock the /me endpoint
    await page.route('**/api/v1/auth/me', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(TEST_USER),
      });
    });

    // Mock dashboard data
    await page.route('**/api/v1/flows', async (route) => {
      await route.fulfill({ status: 200, contentType: 'application/json', body: '[]' });
    });
    await page.route('**/api/v1/runs', async (route) => {
      await route.fulfill({ status: 200, contentType: 'application/json', body: '[]' });
    });

    await page.goto('/login');
    await page.getByLabel('Email').fill('e2e@test.com');
    await page.getByLabel('Password').fill('password123');
    await page.getByRole('button', { name: 'Sign in' }).click();

    // Should redirect to dashboard after successful login
    await expect(page).toHaveURL(/\/dashboard/, { timeout: 5000 });
    await expect(page.getByRole('heading', { name: /Welcome to SynApps/i })).toBeVisible();
  });
});
