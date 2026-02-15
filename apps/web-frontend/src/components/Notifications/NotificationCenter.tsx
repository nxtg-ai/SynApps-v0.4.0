/**
 * NotificationCenter component
 * Manages and displays notifications for workflow events
 */
import React, { useState, useEffect } from 'react';
import { NotificationItem, WorkflowRunStatus } from '../../types';
import webSocketService from '../../services/WebSocketService';
import './Notifications.css';

interface NotificationCenterProps {
  maxNotifications?: number;
}

const NotificationCenter: React.FC<NotificationCenterProps> = ({ 
  maxNotifications = 10 
}) => {
  const [notifications, setNotifications] = useState<NotificationItem[]>([]);
  const [isOpen, setIsOpen] = useState<boolean>(false);
  const [unreadCount, setUnreadCount] = useState<number>(0);
  
  // Subscribe to workflow status notifications
  useEffect(() => {
    const unsubscribe = webSocketService.subscribeToNotifications(
      handleWorkflowStatusNotification
    );
    return unsubscribe;
  }, []);
  
  // Update unread count when notifications change
  useEffect(() => {
    const count = notifications.filter(n => !n.read).length;
    setUnreadCount(count);
  }, [notifications]);
  
  // Handle workflow status notifications
  const handleWorkflowStatusNotification = (status: WorkflowRunStatus) => {
    // Only create notifications for completed workflows
    if (status.status === 'success' || status.status === 'error') {
      addNotification({
        id: `workflow-${status.run_id}`,
        title: status.status === 'success' ? 'Workflow Completed' : 'Workflow Failed',
        message: status.status === 'success' 
          ? `Your workflow "${status.flow_id}" has completed successfully` 
          : `Your workflow "${status.flow_id}" encountered an error: ${status.error}`,
        type: status.status === 'success' ? 'success' : 'error',
        timestamp: Date.now(),
        read: false
      });
    }
  };
  
  // Add a new notification
  const addNotification = (notification: NotificationItem) => {
    setNotifications(prev => {
      // Check if notification with same ID already exists
      const exists = prev.some(n => n.id === notification.id);
      
      if (exists) {
        // Update existing notification
        return prev.map(n => 
          n.id === notification.id ? { ...notification, read: n.read } : n
        );
      } else {
        // Add new notification and limit the number
        const newNotifications = [notification, ...prev];
        
        if (newNotifications.length > maxNotifications) {
          return newNotifications.slice(0, maxNotifications);
        }
        
        return newNotifications;
      }
    });
  };
  
  // Mark all notifications as read
  const markAllAsRead = () => {
    setNotifications(prev => 
      prev.map(n => ({ ...n, read: true }))
    );
  };
  
  // Mark a single notification as read
  const markAsRead = (id: string) => {
    setNotifications(prev => 
      prev.map(n => 
        n.id === id ? { ...n, read: true } : n
      )
    );
  };
  
  // Clear a notification
  const clearNotification = (id: string) => {
    setNotifications(prev => 
      prev.filter(n => n.id !== id)
    );
  };
  
  // Clear all notifications
  const clearAllNotifications = () => {
    setNotifications([]);
  };
  
  // Toggle notification center
  const toggleNotificationCenter = () => {
    setIsOpen(prev => !prev);
    
    // Mark all as read when opening
    if (!isOpen) {
      markAllAsRead();
    }
  };
  
  // Format timestamp
  const formatTime = (timestamp: number): string => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };
  
  return (
    <div className="notification-center">
      <button 
        className="notification-toggle" 
        onClick={toggleNotificationCenter}
      >
        <span className="bell-icon">🔔</span>
        {unreadCount > 0 && (
          <span className="notification-badge">{unreadCount}</span>
        )}
      </button>
      
      {isOpen && (
        <div className="notification-panel">
          <div className="notification-header">
            <h3>Notifications</h3>
            <div className="notification-actions">
              <button className="read-all-button" onClick={markAllAsRead}>
                Mark all as read
              </button>
              <button className="clear-all-button" onClick={clearAllNotifications}>
                Clear all
              </button>
            </div>
          </div>
          
          <div className="notification-list">
            {notifications.length === 0 ? (
              <div className="empty-notifications">
                No notifications to display
              </div>
            ) : (
              notifications.map(notification => (
                <div 
                  key={notification.id} 
                  className={`notification-item ${notification.type} ${notification.read ? 'read' : 'unread'}`}
                  onClick={() => markAsRead(notification.id)}
                >
                  <div className="notification-icon">
                    {notification.type === 'success' && '✅'}
                    {notification.type === 'error' && '❌'}
                    {notification.type === 'info' && 'ℹ️'}
                    {notification.type === 'warning' && '⚠️'}
                  </div>
                  <div className="notification-content">
                    <div className="notification-title">{notification.title}</div>
                    <div className="notification-message">{notification.message}</div>
                    <div className="notification-time">{formatTime(notification.timestamp)}</div>
                  </div>
                  <button 
                    className="notification-clear" 
                    onClick={(e) => {
                      e.stopPropagation();
                      clearNotification(notification.id);
                    }}
                  >
                    ×
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default NotificationCenter;
