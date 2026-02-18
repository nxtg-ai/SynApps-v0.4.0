import { describe, it, expect, vi, beforeEach } from 'vitest';
import { WebSocketService as WebSocketServiceClass } from './WebSocketService'; // Import the class directly
import { WebSocketMessage, WorkflowRunStatus } from '../types';

// Mock the global WebSocket constructor
const mockWebSocket = vi.fn(() => ({
  send: vi.fn(),
  close: vi.fn(),
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
  onopen: vi.fn(),
  onmessage: vi.fn(),
  onclose: vi.fn(),
  onerror: vi.fn(),
}));
vi.stubGlobal('WebSocket', mockWebSocket);

// Mock browser Notification API
const mockNotification = vi.fn();
mockNotification.permission = 'default';
mockNotification.requestPermission = vi.fn(() => Promise.resolve('granted'));
vi.stubGlobal('Notification', mockNotification);

// Mock window.location for protocol
vi.stubGlobal('window', {
  location: {
    protocol: 'http:',
  },
});

// Mock document.addEventListener for Notification permission setup
const originalAddEventListener = document.addEventListener;
const mockDocumentAddEventListener = vi.fn();
vi.stubGlobal('document', {
  addEventListener: mockDocumentAddEventListener,
});

describe('WebSocketService', () => {
  let service: WebSocketServiceClass;

  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();

    // Reset WebSocket mock behavior
    mockWebSocket.mockClear();
    mockWebSocket.mockReturnValue({
      send: vi.fn(),
      close: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      onopen: vi.fn(),
      onmessage: vi.fn(),
      onclose: vi.fn(),
      onerror: vi.fn(),
    });

    // Reset Notification mock behavior
    mockNotification.mockClear();
    mockNotification.permission = 'default';
    mockNotification.requestPermission.mockClear();
    
    // Reset document.addEventListener mock
    mockDocumentAddEventListener.mockClear();
    // Restore original addEventListener to avoid interfering with Vitest's own listeners if any
    document.addEventListener = originalAddEventListener;


    // Set a default value for process.env.REACT_APP_WEBSOCKET_URL
    vi.stubEnv('REACT_APP_WEBSOCKET_URL', 'ws://localhost:8000/ws');

    service = new WebSocketServiceClass();
  });

  afterEach(() => {
    vi.restoreAllMocks(); // Restore mocks for global objects and timers
    vi.useRealTimers();
  });

  it('should initialize without connecting and setup notifications', () => {
    expect(mockWebSocket).not.toHaveBeenCalled();
    expect(service['isConnected']).toBe(false);
    expect(mockDocumentAddEventListener).toHaveBeenCalledWith('click', expect.any(Function), { once: true });
  });

  describe('connect', () => {
    it('should create a new WebSocket connection with the correct URL', () => {
      service.connect();
      expect(mockWebSocket).toHaveBeenCalledTimes(1);
      expect(mockWebSocket).toHaveBeenCalledWith('ws://localhost:8000/ws');
      expect(service['socket']?.onopen).toBeInstanceOf(Function);
      expect(service['socket']?.onmessage).toBeInstanceOf(Function);
      expect(service['socket']?.onclose).toBeInstanceOf(Function);
      expect(service['socket']?.onerror).toBeInstanceOf(Function);
    });

    it('should set isConnected to true on open', () => {
      service.connect();
      const instance = mockWebSocket.mock.results[0].value;
      act(() => {
        instance.onopen();
      });
      expect(service['isConnected']).toBe(true);
      expect(service['reconnectTimer']).toBeNull();
    });

    it('should clear reconnect timer on open', () => {
      service.connect();
      service['reconnectTimer'] = setTimeout(() => {}, 1000); // Simulate an active timer
      const instance = mockWebSocket.mock.results[0].value;
      act(() => {
        instance.onopen();
      });
      expect(service['reconnectTimer']).toBeNull();
    });

    it('should set isConnected to false and schedule reconnect on close', () => {
      service.connect();
      const instance = mockWebSocket.mock.results[0].value;
      act(() => {
        instance.onclose();
      });
      expect(service['isConnected']).toBe(false);
      expect(service['reconnectTimer']).not.toBeNull();
      expect(vi.getTimerCount()).toBe(1); // One timer should be scheduled
    });

    it('should disconnect and schedule reconnect on error', () => {
      service.connect();
      const instance = mockWebSocket.mock.results[0].value;
      const closeSpy = vi.spyOn(instance, 'close');
      
      act(() => {
        instance.onerror(new Event('error'));
      });
      expect(closeSpy).toHaveBeenCalledTimes(1); // Disconnect should be called
      expect(service['reconnectTimer']).not.toBeNull();
      expect(vi.getTimerCount()).toBe(1);
    });
  });

  describe('disconnect', () => {
    it('should close the WebSocket connection', () => {
      service.connect();
      const instance = mockWebSocket.mock.results[0].value;
      const closeSpy = vi.spyOn(instance, 'close');
      
      service.disconnect();
      expect(closeSpy).toHaveBeenCalledTimes(1);
      expect(service['socket']).toBeNull();
      expect(service['isConnected']).toBe(false);
    });
  });

  describe('send', () => {
    it('should send a message when connected', () => {
      service.connect();
      const instance = mockWebSocket.mock.results[0].value;
      const sendSpy = vi.spyOn(instance, 'send');
      
      act(() => {
        instance.onopen(); // Ensure connected
      });

      const message = { type: 'test', data: { value: 123 } };
      service.send(message.type, message.data);
      expect(sendSpy).toHaveBeenCalledWith(JSON.stringify(message));
    });

    it('should connect first then send if disconnected', () => {
      const sendSpy = vi.fn();
      mockWebSocket.mockReturnValue({ // Provide a fresh mock for the second connect call
        send: sendSpy,
        close: vi.fn(),
        onopen: vi.fn(),
        onmessage: vi.fn(),
        onclose: vi.fn(),
        onerror: vi.fn(),
      });
      
      service.send('test', { value: 123 });
      expect(mockWebSocket).toHaveBeenCalledTimes(1); // First call to connect
      expect(sendSpy).not.toHaveBeenCalled(); // Not sent yet as not open

      const instance = mockWebSocket.mock.results[0].value;
      act(() => {
        instance.onopen(); // Open the initial connection
      });
      // After onopen, the send should happen
      expect(sendSpy).toHaveBeenCalledWith(JSON.stringify({ type: 'test', data: { value: 123 } }));
    });
  });

  describe('subscribe', () => {
    it('should register a callback for a message type', () => {
      const callback = vi.fn();
      const unsubscribe = service.subscribe('testType', callback);

      // Simulate a message arriving
      act(() => {
        service['handleMessage']({ type: 'testType', data: 'hello' });
      });
      expect(callback).toHaveBeenCalledWith('hello');

      // Test unsubscribe
      unsubscribe();
      act(() => {
        service['handleMessage']({ type: 'testType', data: 'hello again' });
      });
      expect(callback).toHaveBeenCalledTimes(1); // Should not be called again
    });
  });

  describe('subscribeToNotifications', () => {
    it('should register a callback for workflow notifications', () => {
      const callback = vi.fn();
      const unsubscribe = service.subscribeToNotifications(callback);

      const status: WorkflowRunStatus = { flow_id: '123', status: 'running', progress: 1, total_steps: 10 };
      act(() => {
        service['handleWorkflowStatus'](status);
      });
      expect(callback).toHaveBeenCalledWith(status);

      // Test unsubscribe
      unsubscribe();
      act(() => {
        service['handleWorkflowStatus'](status);
      });
      expect(callback).toHaveBeenCalledTimes(1); // Should not be called again
    });
  });

  describe('handleMessage', () => {
    it('should dispatch messages to correct subscribers', () => {
      const callback1 = vi.fn();
      const callback2 = vi.fn();
      service.subscribe('type1', callback1);
      service.subscribe('type2', callback2);

      act(() => {
        service['handleMessage']({ type: 'type1', data: 'data1' });
      });
      expect(callback1).toHaveBeenCalledWith('data1');
      expect(callback2).not.toHaveBeenCalled();

      act(() => {
        service['handleMessage']({ type: 'type2', data: 'data2' });
      });
      expect(callback1).toHaveBeenCalledTimes(1);
      expect(callback2).toHaveBeenCalledWith('data2');
    });

    it('should call handleWorkflowStatus for workflow.status messages', () => {
      const handleWorkflowStatusSpy = vi.spyOn(service as any, 'handleWorkflowStatus');
      const status: WorkflowRunStatus = { flow_id: '123', status: 'running', progress: 1, total_steps: 10 };
      
      act(() => {
        service['handleMessage']({ type: 'workflow.status', data: status });
      });
      expect(handleWorkflowStatusSpy).toHaveBeenCalledWith(status);
    });
  });

  describe('handleWorkflowStatus', () => {
    it('should notify all registered notification callbacks', () => {
      const callback1 = vi.fn();
      const callback2 = vi.fn();
      service.subscribeToNotifications(callback1);
      service.subscribeToNotifications(callback2);

      const status: WorkflowRunStatus = { flow_id: '123', status: 'completed', progress: 10, total_steps: 10 };
      act(() => {
        service['handleWorkflowStatus'](status);
      });
      expect(callback1).toHaveBeenCalledWith(status);
      expect(callback2).toHaveBeenCalledWith(status);
    });

    it('should send browser notification if status is success or error', () => {
      const sendBrowserNotificationSpy = vi.spyOn(service as any, 'sendBrowserNotification');
      mockNotification.permission = 'granted'; // Grant permission for notification test

      const successStatus: WorkflowRunStatus = { flow_id: '123', status: 'success', progress: 10, total_steps: 10 };
      act(() => {
        service['handleWorkflowStatus'](successStatus);
      });
      expect(sendBrowserNotificationSpy).toHaveBeenCalledWith(successStatus);
      
      sendBrowserNotificationSpy.mockClear();

      const errorStatus: WorkflowRunStatus = { flow_id: '123', status: 'error', progress: 5, total_steps: 10, error: 'test error' };
      act(() => {
        service['handleWorkflowStatus'](errorStatus);
      });
      expect(sendBrowserNotificationSpy).toHaveBeenCalledWith(errorStatus);
    });

    it('should not send browser notification if status is not success or error', () => {
      const sendBrowserNotificationSpy = vi.spyOn(service as any, 'sendBrowserNotification');
      mockNotification.permission = 'granted';

      const runningStatus: WorkflowRunStatus = { flow_id: '123', status: 'running', progress: 5, total_steps: 10 };
      act(() => {
        service['handleWorkflowStatus'](runningStatus);
      });
      expect(sendBrowserNotificationSpy).not.toHaveBeenCalled();
    });
  });

  describe('sendBrowserNotification', () => {
    it('should create a browser notification if permission is granted and status is success', () => {
      mockNotification.permission = 'granted';
      const status: WorkflowRunStatus = { flow_id: '123', status: 'success', progress: 10, total_steps: 10 };
      
      act(() => {
        service['sendBrowserNotification'](status);
      });
      expect(mockNotification).toHaveBeenCalledTimes(1);
      expect(mockNotification).toHaveBeenCalledWith('Workflow completed', expect.any(Object));
    });

    it('should create a browser notification if permission is granted and status is error', () => {
      mockNotification.permission = 'granted';
      const status: WorkflowRunStatus = { flow_id: '123', status: 'error', progress: 5, total_steps: 10, error: 'test error' };
      
      act(() => {
        service['sendBrowserNotification'](status);
      });
      expect(mockNotification).toHaveBeenCalledTimes(1);
      expect(mockNotification).toHaveBeenCalledWith('Workflow failed', expect.any(Object));
    });

    it('should not create a browser notification if permission is not granted', () => {
      mockNotification.permission = 'denied';
      const status: WorkflowRunStatus = { flow_id: '123', status: 'success', progress: 10, total_steps: 10 };
      
      act(() => {
        service['sendBrowserNotification'](status);
      });
      expect(mockNotification).not.toHaveBeenCalled();
    });
  });

  describe('scheduleReconnect', () => {
    it('should schedule a reconnect attempt after 3 seconds', () => {
      service['scheduleReconnect']();
      expect(vi.getTimerCount()).toBe(1);
      vi.advanceTimersByTime(3000);
      expect(service['connect']).toHaveBeenCalledTimes(1); // Assuming connect is spied on or mocked
    });

    it('should not schedule multiple reconnects', () => {
      service['scheduleReconnect']();
      service['scheduleReconnect']();
      expect(vi.getTimerCount()).toBe(1);
    });
  });
});
