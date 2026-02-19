import { beforeEach, describe, expect, it, vi, afterEach } from 'vitest';
import type { WorkflowRunStatus } from '../types';

class MockWebSocket {
  static instances: MockWebSocket[] = [];

  public readonly url: string;
  public onopen: ((event: Event) => void) | null = null;
  public onmessage: ((event: MessageEvent) => void) | null = null;
  public onclose: ((event: CloseEvent) => void) | null = null;
  public onerror: ((event: Event) => void) | null = null;
  public send = vi.fn();
  public close = vi.fn();

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  emitOpen() {
    this.onopen?.(new Event('open'));
  }

  emitMessage(data: unknown) {
    this.onmessage?.({ data: JSON.stringify(data) } as MessageEvent);
  }

  emitRawMessage(raw: string) {
    this.onmessage?.({ data: raw } as MessageEvent);
  }

  emitClose() {
    this.onclose?.(new CloseEvent('close'));
  }

  emitError() {
    this.onerror?.(new Event('error'));
  }
}

const loadService = async () => {
  vi.resetModules();
  const module = await import('./WebSocketService');
  return module.default;
};

describe('WebSocketService', () => {
  let notificationConstructor: ReturnType<typeof vi.fn>;
  let addEventListenerSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    vi.useFakeTimers();
    vi.clearAllMocks();
    vi.unstubAllEnvs();
    MockWebSocket.instances = [];

    notificationConstructor = vi.fn();
    (notificationConstructor as unknown as { permission: string }).permission = 'default';
    (
      notificationConstructor as unknown as {
        requestPermission: () => Promise<NotificationPermission>;
      }
    ).requestPermission = vi.fn().mockResolvedValue('granted');

    vi.stubGlobal('WebSocket', MockWebSocket as unknown as typeof WebSocket);
    vi.stubGlobal('Notification', notificationConstructor as unknown as typeof Notification);

    addEventListenerSpy = vi.spyOn(document, 'addEventListener');
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('sets up browser notification permission prompt on initialization', async () => {
    await loadService();

    expect(addEventListenerSpy).toHaveBeenCalledWith('click', expect.any(Function), { once: true });

    const clickHandler = addEventListenerSpy.mock.calls.find(([eventName]) => eventName === 'click')?.[1] as
      | (() => void)
      | undefined;

    clickHandler?.();

    expect(
      (
        notificationConstructor as unknown as {
          requestPermission: ReturnType<typeof vi.fn>;
        }
      ).requestPermission,
    ).toHaveBeenCalledTimes(1);
  });

  it('connects using REACT_APP_WEBSOCKET_URL and sends messages when connected', async () => {
    vi.stubEnv('REACT_APP_WEBSOCKET_URL', 'ws://example.test/ws');
    const service = await loadService();

    service.connect();
    expect(MockWebSocket.instances).toHaveLength(1);
    expect(MockWebSocket.instances[0].url).toBe('ws://example.test/ws');

    MockWebSocket.instances[0].emitOpen();
    service.send('hello', { value: 1 });

    expect(MockWebSocket.instances[0].send).toHaveBeenCalledWith(
      JSON.stringify({ type: 'hello', data: { value: 1 } }),
    );
  });

  it('warns and reconnects when send is called while disconnected', async () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const service = await loadService();

    service.send('queued', { retry: true });

    expect(warnSpy).toHaveBeenCalledWith('Tried to send message while disconnected');
    expect(MockWebSocket.instances).toHaveLength(1);
  });

  it('routes messages by type and supports unsubscribe', async () => {
    const service = await loadService();
    const callback = vi.fn();
    const unsubscribe = service.subscribe('custom.event', callback);

    service.connect();
    MockWebSocket.instances[0].emitOpen();
    MockWebSocket.instances[0].emitMessage({
      type: 'custom.event',
      data: { ok: true },
    });

    expect(callback).toHaveBeenCalledWith({ ok: true });

    unsubscribe();
    MockWebSocket.instances[0].emitMessage({
      type: 'custom.event',
      data: { ok: false },
    });

    expect(callback).toHaveBeenCalledTimes(1);
  });

  it('handles workflow.status notifications and browser notifications', async () => {
    (notificationConstructor as unknown as { permission: string }).permission = 'granted';
    const service = await loadService();
    const notificationCallback = vi.fn();
    service.subscribeToNotifications(notificationCallback);

    const successStatus: WorkflowRunStatus = {
      run_id: 'run-1',
      flow_id: 'flow-1',
      status: 'success',
      progress: 1,
      total_steps: 1,
      start_time: 1,
      end_time: 2,
      error: '',
      results: {},
    };

    const errorStatus: WorkflowRunStatus = {
      run_id: 'run-2',
      flow_id: 'flow-1',
      status: 'error',
      progress: 1,
      total_steps: 1,
      start_time: 1,
      end_time: 2,
      error: 'failed',
      results: {},
    };

    service.connect();
    MockWebSocket.instances[0].emitOpen();
    MockWebSocket.instances[0].emitMessage({ type: 'workflow.status', data: successStatus });
    MockWebSocket.instances[0].emitMessage({ type: 'workflow.status', data: errorStatus });

    expect(notificationCallback).toHaveBeenCalledWith(successStatus);
    expect(notificationCallback).toHaveBeenCalledWith(errorStatus);
    expect(notificationConstructor).toHaveBeenCalledTimes(2);
    expect(notificationConstructor).toHaveBeenNthCalledWith(
      1,
      'Workflow completed',
      expect.objectContaining({
        body: 'Your workflow "flow-1" has completed successfully',
      }),
    );
    expect(notificationConstructor).toHaveBeenNthCalledWith(
      2,
      'Workflow failed',
      expect.objectContaining({
        body: 'Your workflow "flow-1" encountered an error: failed',
      }),
    );
  });

  it('logs parse errors for invalid websocket payloads', async () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const service = await loadService();

    service.connect();
    MockWebSocket.instances[0].emitRawMessage('{not-json');

    expect(errorSpy).toHaveBeenCalledWith('Failed to parse WebSocket message', expect.any(SyntaxError));
  });

  it('disconnects existing socket when connect is called again', async () => {
    const service = await loadService();

    service.connect();
    const firstSocket = MockWebSocket.instances[0];
    service.connect();

    expect(firstSocket.close).toHaveBeenCalledTimes(1);
    expect(MockWebSocket.instances).toHaveLength(2);
  });

  it('schedules reconnect on close and on error', async () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const service = await loadService();

    service.connect();
    const socket = MockWebSocket.instances[0];
    socket.emitClose();
    vi.advanceTimersByTime(3000);
    expect(MockWebSocket.instances).toHaveLength(2);

    const reconnectSocket = MockWebSocket.instances[1];
    reconnectSocket.emitOpen();
    reconnectSocket.emitError();
    expect(errorSpy).toHaveBeenCalledWith('WebSocket error', expect.any(Event));
    expect(reconnectSocket.close).toHaveBeenCalledTimes(1);

    vi.advanceTimersByTime(3000);
    expect(MockWebSocket.instances).toHaveLength(3);
  });
});
