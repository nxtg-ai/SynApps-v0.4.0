import { expect, test } from '@playwright/test';

const dashboardFlows = [
  {
    id: 'flow-existing',
    name: 'Existing Flow',
    nodes: [{ id: 'start', type: 'start', position: { x: 0, y: 0 }, data: { label: 'Start' } }],
    edges: [],
  },
];

const applets = [
  {
    type: 'writer',
    name: 'WriterApplet',
    description: 'Writes text',
    version: '1.0.0',
    capabilities: ['draft', 'rewrite'],
  },
  {
    type: 'artist',
    name: 'ArtistApplet',
    description: 'Creates images',
    version: '1.0.0',
    capabilities: ['image'],
  },
];

const savedFlow = {
  id: 'flow-created',
  name: 'Blog Post Writer',
  nodes: [
    { id: 'start', type: 'start', position: { x: 250, y: 25 }, data: { label: 'Start' } },
    { id: 'writer1', type: 'writer', position: { x: 250, y: 125 }, data: { label: 'Draft Writer' } },
    { id: 'end', type: 'end', position: { x: 250, y: 225 }, data: { label: 'End' } },
  ],
  edges: [
    { id: 'start-writer1', source: 'start', target: 'writer1', animated: false },
    { id: 'writer1-end', source: 'writer1', target: 'end', animated: false },
  ],
};

test.beforeEach(async ({ page }) => {
  await page.route('**/flows', async (route) => {
    if (route.request().method() === 'GET') {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(dashboardFlows) });
      return;
    }

    if (route.request().method() === 'POST') {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ id: 'flow-created' }) });
      return;
    }

    await route.fallback();
  });

  await page.route('**/flows/**', async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;

    if (request.method() === 'GET' && path.endsWith('/flows/flow-created')) {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(savedFlow) });
      return;
    }

    if (request.method() === 'POST' && path.endsWith('/flows/flow-created/run')) {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ run_id: 'run-1' }) });
      return;
    }

    await route.fallback();
  });

  await page.route('**/runs', async (route) => {
    if (route.request().method() === 'GET') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          {
            run_id: 'run-1',
            flow_id: 'flow-existing',
            status: 'success',
            progress: 1,
            total_steps: 1,
            start_time: 1700000000,
            end_time: 1700000001,
            results: {},
          },
        ]),
      });
      return;
    }

    await route.fallback();
  });

  await page.route('**/applets', async (route) => {
    if (route.request().method() === 'GET') {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(applets) });
      return;
    }

    await route.fallback();
  });
});

test.describe('Frontend happy-path', () => {
  test('navigates between dashboard, editor, and applet library', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveURL(/\/dashboard$/);
    await expect(page.getByRole('heading', { level: 1, name: 'Dashboard' })).toBeVisible();

    await page.getByRole('link', { name: 'Applet Library' }).click();
    await expect(page).toHaveURL(/\/applets$/);
    await expect(page.getByRole('heading', { level: 1, name: 'Applet Library' })).toBeVisible();

    await page.getByRole('button', { name: 'Use in Workflow' }).click();
    await expect(page).toHaveURL(/\/editor$/);
    await expect(page.getByRole('heading', { level: 1, name: 'Workflow Editor' })).toBeVisible();
  });

  test('creates a new workflow from template and lands in editor', async ({ page }) => {
    await page.goto('/dashboard');

    await page.getByRole('button', { name: 'Create New Workflow' }).click();
    await expect(page.getByRole('heading', { level: 2, name: 'Choose a Template' })).toBeVisible();

    await page.getByRole('heading', { level: 3, name: 'Blog Post Writer' }).click();
    await page.getByRole('button', { name: 'Create Flow' }).click();

    await expect(page).toHaveURL(/\/editor\/flow-created$/);
    await expect(page.locator('.workflow-name-input')).toHaveValue('Blog Post Writer');
    await expect(page.getByText('Available Nodes')).toBeVisible();
  });

  test('runs an existing workflow from editor', async ({ page }) => {
    await page.goto('/editor/flow-created');

    await expect(page.getByRole('button', { name: 'Run Workflow' })).toBeEnabled();
    await page.locator('.input-textarea').fill('{"topic":"release notes"}');
    await page.getByRole('button', { name: 'Run Workflow' }).click();
    await expect(page.getByRole('button', { name: 'Run Workflow' })).toBeEnabled();
  });

  test('updates and persists settings locally', async ({ page }) => {
    page.on('dialog', async (dialog) => {
      await dialog.accept();
    });

    await page.goto('/settings');
    await page.locator('#openai_key').fill('sk-test-openai-key');
    await page.locator('#dark_mode').check();
    await page.getByRole('button', { name: 'Save Settings' }).click();

    await page.reload();
    await expect(page.locator('#openai_key')).toHaveValue('sk-test-openai-key');
    await expect(page.locator('#dark_mode')).toBeChecked();
  });
});
