/**
 * NotFoundPage component
 * Displayed when a user navigates to an invalid route
 */
import React from 'react';
import { Link } from 'react-router-dom';
import MainLayout from '../../components/Layout/MainLayout';
import './NotFoundPage.css';

const NotFoundPage: React.FC = () => {
  return (
    <MainLayout title="Page Not Found">
      <div className="not-found-page">
        <div className="not-found-content">
          <div className="not-found-icon">
            🤖
          </div>
          <h2>404 - Page Not Found</h2>
          <p>
            The page you're looking for doesn't exist or has been moved.
          </p>
          <div className="action-buttons">
            <Link to="/" className="home-button">
              Go to Dashboard
            </Link>
            <Link to="/editor" className="editor-button">
              Create a Workflow
            </Link>
          </div>
        </div>
      </div>
    </MainLayout>
  );
};

export default NotFoundPage;
