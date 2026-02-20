/**
 * WebSocketService - Handles real-time communication with the backend
 */
import { WebSocketMessage, WorkflowRunStatus } from '../types';

// For browser notifications
type NotificationCallback = (status: WorkflowRunStatus) => void;

class WebSocketService {
  private socket: WebSocket | null = null;
  private isConnected: boolean = false;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private callbacks: Map<string, Set<(data: any) => void>> = new Map();
  private notificationCallbacks: Set<NotificationCallback> = new Set();
  
  constructor() {
    this.setupNotifications();
  }
  
  /**
   * Connect to the WebSocket server
   */
  public connect(): void {
    if (this.socket) {
      this.disconnect();
    }
    
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    // Make sure to use the correct backend port (8000)
    const wsUrl = process.env.REACT_APP_WEBSOCKET_URL || `${protocol}://localhost:8000/ws`;

    
    this.socket = new WebSocket(wsUrl);
    
    this.socket.onopen = () => {
      this.isConnected = true;
      if (this.reconnectTimer) {
        clearTimeout(this.reconnectTimer);
        this.reconnectTimer = null;
      }

      // Send auth token if available
      const token =
        typeof window !== 'undefined' ? window.localStorage.getItem('access_token') : null;
      if (token && this.socket) {
        this.socket.send(JSON.stringify({ type: 'auth', token }));
      }
    };
    
    this.socket.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data) as WebSocketMessage;
        this.handleMessage(message);
      } catch (error) {
        console.error('Failed to parse WebSocket message', error);
      }
    };
    
    this.socket.onclose = () => {

      this.isConnected = false;
      this.scheduleReconnect();
    };
    
    this.socket.onerror = (error) => {
      console.error('WebSocket error', error);
      this.disconnect();
      this.scheduleReconnect();
    };
  }
  
  /**
   * Disconnect from the WebSocket server
   */
  public disconnect(): void {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
      this.isConnected = false;
    }
  }
  
  /**
   * Send a message to the WebSocket server
   */
  public send(type: string, data: any): void {
    if (this.isConnected && this.socket) {
      const message: WebSocketMessage = { type, data };
      this.socket.send(JSON.stringify(message));
    } else {
      console.warn('Tried to send message while disconnected');
      this.connect();
    }
  }
  
  /**
   * Subscribe to a specific message type
   */
  public subscribe(type: string, callback: (data: any) => void): () => void {
    if (!this.callbacks.has(type)) {
      this.callbacks.set(type, new Set());
    }
    
    this.callbacks.get(type)!.add(callback);
    
    // Return unsubscribe function
    return () => {
      const callbacks = this.callbacks.get(type);
      if (callbacks) {
        callbacks.delete(callback);
        if (callbacks.size === 0) {
          this.callbacks.delete(type);
        }
      }
    };
  }
  
  /**
   * Subscribe to workflow notifications
   */
  public subscribeToNotifications(callback: NotificationCallback): () => void {
    this.notificationCallbacks.add(callback);
    
    // Return unsubscribe function
    return () => {
      this.notificationCallbacks.delete(callback);
    };
  }
  
  /**
   * Handle incoming WebSocket messages
   */
  private handleMessage(message: WebSocketMessage): void {
    const { type, data } = message;
    
    // Call all registered callbacks for this type
    const callbacks = this.callbacks.get(type);
    if (callbacks) {
      callbacks.forEach(callback => callback(data));
    }
    
    // Special handling for workflow status messages
    if (type === 'workflow.status') {
      this.handleWorkflowStatus(data as WorkflowRunStatus);
    }
  }
  
  /**
   * Handle workflow status messages
   */
  private handleWorkflowStatus(status: WorkflowRunStatus): void {
    // Notify all registered notification callbacks
    this.notificationCallbacks.forEach(callback => callback(status));
    
    // Send browser notification if workflow completed
    if (status.status === 'success' || status.status === 'error') {
      this.sendBrowserNotification(status);
    }
  }
  
  /**
   * Send a browser notification
   */
  private sendBrowserNotification(status: WorkflowRunStatus): void {
    if (Notification.permission === 'granted') {
      const title = status.status === 'success' ? 'Workflow completed' : 'Workflow failed';
      const options = {
        body: status.status === 'success' 
          ? `Your workflow "${status.flow_id}" has completed successfully` 
          : `Your workflow "${status.flow_id}" encountered an error: ${status.error}`,
        icon: '/logo.png'
      };
      
      new Notification(title, options);
    }
  }
  
  /**
   * Set up browser notifications
   */
  private setupNotifications(): void {
    if ('Notification' in window) {
      if (Notification.permission !== 'granted' && Notification.permission !== 'denied') {
        // We'll request permission when user interacts with the page
        document.addEventListener('click', () => {
          Notification.requestPermission();
        }, { once: true });
      }
    }
  }
  
  /**
   * Schedule reconnection after disconnection
   */
  private scheduleReconnect(): void {
    if (!this.reconnectTimer) {
      this.reconnectTimer = setTimeout(() => {
        this.connect();
      }, 3000);
    }
  }
}

// Create a singleton instance
export const webSocketService = new WebSocketService();
export default webSocketService;
