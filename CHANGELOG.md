# Changelog

All notable changes to the SynApps project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.2] - 2025-06-10

### Added

- **Workflow Editor Enhancements**
  - Added node deletion functionality via keyboard (Delete key) and context menu
  - Implemented proper cleanup of connected edges when deleting nodes
  - Added enhanced context menu for workflow nodes with the following features:
    - Direct access to node configuration
    - Improved positioning next to nodes
    - Better styling with icons and dividers

## [0.5.1] - 2025-06-09

### Fixed

- **Workflow History Page**
  - Fixed run selection to show the latest workflow run by default instead of the oldest run
  - Ensured proper sorting of workflow runs by timestamp

## [0.5.0] - 2025-06-08

### Added

- **Node-Specific Configuration**
  - Implemented `NodeConfig` component for node-specific input configuration
  - Added configuration panels for Start, Writer, and Artist nodes:
    - **Start Node**: Input text area for initial workflow input data with JSON parsing
    - **Writer Node**: System prompt text area for configuring language model prompts
    - **Artist Node**: System prompt text area and generator selection dropdown
  - Added gear icon toggle button to open/close node configuration panels

- **Dynamic Workflow Execution Feedback**
  - Added status indicators (colored dots) in the upper-right corner of each node
  - Implemented progressive lighting of status indicators as workflow steps complete
  - Added visual feedback for running, success, and error states with appropriate colors
  - Enhanced node border styling to reflect execution status
  - Added backend tracking of completed workflow nodes via database
  - Implemented real-time WebSocket broadcasting of node completion status

### Changed

- **Workflow Editor UI**
  - Removed global Input Data Panel in favor of node-specific configurations
  - Updated CSS styling for better layout and consistency
  
- **Orchestrator Backend**
  - Enhanced `_execute_flow_async` to handle node-specific configuration data
  - Updated flow execution to use Start node's input data configuration
  - Added support for passing node-specific metadata (system prompts, generator selection) to applets
  - Added tracking of completed nodes during workflow execution
  - Created database migration for storing completed nodes in workflow runs
  - Enhanced WebSocket status updates to include completed nodes information

## [0.4.0] - 2025-06-07

### Added

- **Enhanced Run History Page**
  - Added workflow names to the Run History page for better identification
  - Implemented sorting functionality with newest-first as default
  - Added toggle button to switch between newest-first and oldest-first sorting
  - Improved UI styling for better readability

### Changed

- **FastAPI Modernization**
  - Replaced deprecated lifecycle event handlers (`@app.on_event`) with the modern lifespan context manager
  - Created a helper function `model_to_dict` to abstract Pydantic model serialization differences
  - Updated all Pydantic model dictionary conversions for compatibility with both v1 and v2
  - Fixed dependency injection in the Orchestrator class

## [0.3.0] - 2025-06-06

### Added

- **Database Persistence**
  - Implemented SQLAlchemy async ORM for database operations
  - Created models for flows, nodes, edges, and workflow runs
  - Added Alembic migrations for database schema management
  - Implemented repository pattern for data access

### Fixed

- **Workflow Execution Issues**
  - Consolidated duplicate `Orchestrator` class definitions
  - Fixed inconsistent handling of Pydantic models vs dictionaries
  - Ensured proper dictionary access for status objects
  - Fixed the `load_applet` method to properly load applet modules

- **Async Function Handling**
  - Properly awaited coroutines for database initialization and connection closing
  - Fixed async patterns in workflow execution
  - Improved error handling during async operations

## [0.2.0] - 2025-05-31

### Fixed

- **Workflow Editor Infinite Update Loop**
  - Modified the `handleNodeClick` function in `EditorPage.tsx` to check if the node already exists in the workflow
  - Used `useRef` to track previous flow values in `WorkflowCanvas.tsx` to prevent unnecessary re-renders
  - Implemented proper state comparison using JSON stringification in the `handleFlowChange` function

- **Node Panel Not Displaying**
  - Restructured the editor sidebar layout in `EditorPage.tsx` to properly display all panels
  - Added scrolling to the sidebar in `EditorPage.css` to ensure all content is accessible
  - Removed duplicate node panel code from the bottom of the component

- **Drag and Drop Not Working**
  - Implemented proper `onDrop` and `onDragOver` handlers in the `WorkflowCanvas` component
  - Fixed the `handleNodeClick` function to properly add new nodes when clicked
  - Added proper TypeScript type checking to prevent null reference errors

### Added

- **Enhanced Sidebar Layout**
  - Improved organization of the sidebar with Input Panel, Available Nodes, and Results Panel
  - Added overflow scrolling to ensure all panels are accessible

### Changed

- **Node Addition Logic**
  - Modified how nodes are added to the workflow when clicked vs. dragged
  - Ensured proper flow state updates when adding new nodes

## [0.1.0] - 2025-05-22

### Added

- **Initial MVP Release**
  - Basic workflow editor with React Flow
  - Support for Writer, Artist, and Memory applets
  - Real-time workflow execution via WebSockets
  - Results panel with text and image output support

### Fixed

- **Image Results Rendering**
  - Implemented a recursive scanning algorithm to detect image data in various formats
  - Added support for multiple image formats (URLs, base64, nested objects)

- **WebSocket Connection Issues**
  - Updated WebSocket URL to explicitly use `localhost:8000`
  - Added better connection management and logging
  - Implemented proper subscription in the EditorPage component
