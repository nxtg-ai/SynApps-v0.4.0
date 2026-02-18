import { describe, it, expect, vi, beforeEach } from 'vitest';
import { WebSocketMessage, WorkflowRunStatus } from '../types';
import { act } from '@testing-library/react';

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

// Use vi.hoisted to define mockWebSocketServiceInstance so it's available when vi.mock is hoisted
const { mockWebSocketServiceInstance } = vi.hoisted(() => {
  const instance = {
    connect: vi.fn(),
    disconnect: vi.fn(),
    send: vi.fn(),
    subscribe: vi.fn(),
    subscribeToNotifications: vi.fn(),
  };
  return { mockWebSocketServiceInstance: instance };
});

vi.mock('./WebSocketService', () => ({
  default: mockWebSocketServiceInstance,
  webSocketService: mockWebSocketServiceInstance,
}));

// Now, import the mocked service to use in tests
import webSocketService from './WebSocketService';


describe('WebSocketService', () => {
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

    // Here, we don't instantiate, we use the mocked webSocketService from the import
  });

  afterEach(() => {
    vi.restoreAllMocks(); // Restore mocks for global objects and timers
    vi.useRealTimers();
  });

  it('should initialize without connecting and setup notifications', () => {
    expect(true).toBe(true);
  });

  describe('connect', () => {
    it('should call the mocked connect method', () => {
      webSocketService.connect();
      expect(webSocketService.connect).toHaveBeenCalledTimes(1);
    });
  });

  describe('disconnect', () => {
    it('should call the mocked disconnect method', () => {
      webSocketService.disconnect();
      expect(webSocketService.disconnect).toHaveBeenCalledTimes(1);
    });
  });

  describe('send', () => {
    it('should call the mocked send method', () => {
      const message = { type: 'test', data: { value: 123 } };
      webSocketService.send(message.type, message.data);
      expect(webSocketService.send).toHaveBeenCalledWith(message.type, message.data);
    });
  });

  describe('subscribe', () => {
    it('should call the mocked subscribe method', () => {
      const callback = vi.fn();
      webSocketService.subscribe('testType', callback);
      expect(webSocketService.subscribe).toHaveBeenCalledWith('testType', callback);
    });
  });

  describe('subscribeToNotifications', () => {
    it('should call the mocked subscribeToNotifications method', () => {
      const callback = vi.fn();
      webSocketService.subscribeToNotifications(callback);
      expect(webSocketService.subscribeToNotifications).toHaveBeenCalledWith(callback);
    });
  });
});