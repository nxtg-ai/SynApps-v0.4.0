# SynApps v0.4.0 Development Context Primer

This document captures the development context, issues fixed, and guidance for the SynApps v0.4.0 project as of June 8, 2025.

## Project Overview

SynApps is a web-based visual platform for modular AI agents. It allows users to build autonomous AI applets like LEGO blocks, with each applet providing a specialized skill (Writer, Memory, Artist, etc.). A lightweight orchestrator routes messages between these applets, enabling them to collaborate on complex tasks.

### Key Components

1. **Backend (Orchestrator)**
   - Python/FastAPI-based microkernel
   - WebSocket for real-time updates
   - Applet management and workflow execution

2. **Frontend (React Application)**
   - React Flow for workflow visualization
   - Real-time status updates via WebSockets
   - Visual workflow editor

3. **Applets**
   - Writer: Text generation using GPT-4
   - Artist: Image generation using Stable Diffusion
   - Memory: Context storage and retrieval

## Development Environment Setup

### Backend Setup

1. Navigate to the orchestrator directory:
   ```bash
   cd apps/orchestrator
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
4. Initialize the database:
   ```bash
   alembic upgrade head
   ```

4. Run the development server:
   ```bash
   uvicorn main:app --reload
   ```
   
   The backend server will run on http://localhost:8000.

### Frontend Setup

1. Navigate to the web-frontend directory:
   ```bash
   cd apps/web-frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Run the development server:
   ```bash
   npm start
   ```
   
   The frontend will run on http://localhost:3000.

### Environment Variables

Create a `.env` file in the project root with the following:

```
OPENAI_API_KEY=your_openai_api_key
STABILITY_API_KEY=your_stability_api_key
DATABASE_URL=sqlite+aiosqlite:///synapps.db
```

## Issues Fixed

### Backend Issues

1. **Import Path Problem**
   - **Issue**: The orchestrator couldn't find the applet modules (`No module named 'apps'`).
   - **Fix**: Added the project root to the Python path in `main.py`:
     ```python
     import sys
     sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
     ```

2. **Running from Wrong Directory**
   - **Issue**: `uvicorn main:app --reload` failed when run from the wrong directory.
   - **Fix**: Always run the backend server from the `apps/orchestrator` directory.
   
3. **Workflow Execution Errors**
   - **Issue**: Workflow execution was stuck at 0/5 steps due to inconsistent handling of status objects.
   - **Fix**: 
     - Consolidated duplicate `Orchestrator` class definitions
     - Fixed inconsistent handling of Pydantic models vs dictionaries
     - Ensured proper dictionary access for status objects
     - Fixed the `load_applet` method to properly load applet modules
     
4. **Database Persistence Issues**
   - **Issue**: Workflows and their status were not being persisted between server restarts.
   - **Fix**:
     - Implemented SQLAlchemy async ORM models
     - Added Alembic migrations for database schema management
     - Created repositories for database operations
     - Ensured proper initialization and closing of database connections

### Frontend Issues

1. **Module Resolution**
   - **Issue**: Frontend build failed with "Can't resolve './App' in src" error.
   - **Fix**: 
     - Created a proper `tsconfig.json` file
     - Fixed import statements in `index.tsx`

2. **ReactFlow/Zustand Error**
   - **Issue**: "Seems like you have not used zustand provider as an ancestor" error.
   - **Fix**: Wrapped the WorkflowCanvas component with ReactFlowProvider:
     ```tsx
     const WorkflowCanvasWithProvider: React.FC<WorkflowCanvasProps> = (props) => {
       return (
         <ReactFlowProvider>
           <WorkflowCanvas {...props} />
         </ReactFlowProvider>
       );
     };
     ```

3. **WebSocket Connection**
   - **Issue**: WebSocket wasn't connecting properly, leading to "No connected clients to broadcast to" errors.
   - **Fix**: 
     - Updated WebSocket URL to explicitly use `localhost:8000`
     - Added better connection management and logging
     - Implemented proper subscription in the EditorPage component

4. **Workflow Results Not Displaying**
    - **Issue**: Workflow executed successfully but results weren't displayed in the UI.
    - **Fix**:
      - Added state management for workflow results
      - Created a results panel to display text and image outputs
      - Added proper styling for the results section

5. **Image Results Not Rendering**
    - **Issue**: Images from the Artist applet weren't consistently displaying in the results panel.
    - **Fix**:
      - Implemented a robust image detection algorithm that recursively scans result data
      - Added support for multiple image formats (URLs, base64, nested objects)
      - Created better error handling and debugging information

6. **Workflow Editor Infinite Update Loop**
    - **Issue**: Clicking on nodes in the workflow editor caused an infinite update loop.
    - **Fix**:
      - Modified the `handleNodeClick` function to check if the node already exists in the workflow
      - Used `useRef` to track previous flow values and prevent unnecessary re-renders
      - Implemented proper state comparison using JSON stringification

7. **Node Panel Not Displaying**
    - **Issue**: The 'Available Nodes' section was not visible in the workflow editor.
    - **Fix**:
      - Restructured the editor sidebar layout to properly display all panels
      - Added scrolling to the sidebar to ensure all content is accessible
      - Removed duplicate node panel code

8. **Drag and Drop Not Working**
    - **Issue**: Unable to add new nodes to the workflow via drag and drop or clicking.
    - **Fix**:
      - Implemented proper onDrop and onDragOver handlers in the WorkflowCanvas component
      - Fixed the handleNodeClick function to properly add new nodes when clicked
      - Added proper TypeScript type checking to prevent null reference errors

## Recent Improvements

### Backend Enhancements

1. **Database Persistence**:
   - Implemented SQLAlchemy async ORM for database operations
   - Created models for flows, nodes, edges, and workflow runs
   - Added Alembic migrations for database schema management
   - Implemented repository pattern for data access

2. **Workflow Execution**:
   - Fixed workflow status tracking and updates
   - Improved error handling during applet execution
   - Added proper async/await patterns for database operations
   - Fixed applet loading mechanism
   - Enhanced orchestrator to handle node-specific configuration data
   - Updated flow execution to use Start node input data and node-specific settings

3. **API Improvements**:
   - Enhanced WebSocket broadcasting for real-time updates
   - Added proper error responses for API endpoints
   - Improved status tracking and reporting
   - Implemented workflow node completion tracking via completed_applets column

### UI Enhancements

1. **Node-Specific Configuration**:
   - Implemented modular `NodeConfig` component for node-specific input configuration
   - Added configuration panels for Start, Writer, and Artist nodes
   - Created toggle button (gear icon) to open/close configuration panels
   - Removed global Input Data Panel in favor of node-specific configurations
   - Styled configuration panels with consistent design per node type

2. **Reorganized Output Display**:
   - Created a structured, section-based layout for workflow results
   - Separated output into three distinct panels: Text Output, Raw Results, and Image Results
   - Added clear section headers and consistent styling
   - Implemented collapsible sections for better space management

3. **Image Generator Selection**:
   - Moved image generator selection to Artist node configuration
   - Implemented support for both Stability.ai and DALL-E 3
   - Added descriptive labels to explain the strengths of each option

3. **Advanced Image Rendering**:
   - Created a recursive scanning algorithm to locate image data anywhere in the result object
   - Added support for multiple image formats (URLs, base64 encoded strings, nested objects)
   - Implemented source attribution to show where image data was found
   - Added comprehensive logging for debugging image rendering issues

4. **Error States and Feedback**:
   - Improved empty state messaging when data is not available
   - Added helpful debug information directly in the UI
   - Enhanced console logging for troubleshooting
   - Created a more user-friendly loading indicator during workflow execution
   - Implemented dynamic node status indicators showing execution progress in real-time

## Known Issues

1. **ESLint Warnings**: There are several ESLint warnings related to unused variables and dependencies. These don't affect functionality but should be addressed for code quality.

2. **Deprecated Dependencies**: The npm install shows warnings about deprecated packages which should be updated in the future.

3. **Node Positioning**: When adding nodes by clicking (rather than dragging), they appear at a fixed position (100, 100) which may cause overlapping. A more intelligent positioning algorithm could be implemented.

4. **FastAPI Deprecation Warnings**: The FastAPI event handlers use the deprecated `on_event` approach instead of the newer lifespan events.

5. **Pydantic Deprecation Warnings**: The code uses the `.dict()` method which is deprecated in Pydantic v2 in favor of `.model_dump()`.

## Next Steps

### Immediate Tasks

1. **Address ESLint Warnings**: Clean up unused imports and fix missing dependencies in useEffect hooks.

2. **Upgrade Dependencies**: Update deprecated packages to their newer versions.

3. **Error Handling**: Improve error handling for API calls and WebSocket connections.

4. **Update FastAPI Event Handlers**: Replace deprecated `on_event` with lifespan events.

5. **Update Pydantic Methods**: Replace deprecated `.dict()` with `.model_dump()` for Pydantic v2 compatibility.

### Future Enhancements

1. **User Authentication**: Implement user authentication and authorization.

2. **More Applets**: Add more specialized applets (e.g., Researcher, Analyzer, Summarizer).

3. **Workflow Templates**: Create more pre-built workflow templates for common use cases.

4. **Applet Marketplace**: Create a marketplace for users to share and discover applets.

5. **Authentication**: Implement user authentication and authorization.

## Development Workflow Tips

1. **Backend Changes**: After making changes to the backend, restart the uvicorn server.

2. **Frontend Development**: The React dev server supports hot reloading, so most changes appear instantly.

3. **WebSocket Debugging**: Use the browser console to debug WebSocket connections. Look for connection logs and message events.

4. **API Testing**: Use the FastAPI Swagger UI at http://localhost:8000/docs to test API endpoints.

## Troubleshooting

1. **Backend Won't Start**: Ensure you're in the correct directory (`apps/orchestrator`) and the virtual environment is activated.

2. **Frontend Build Errors**: Check for TypeScript errors and ensure all dependencies are installed.

3. **Workflow Execution Fails**: Check the backend logs for errors. Common issues include missing API keys, import errors, or database connection issues.

4. **WebSocket Disconnects**: Ensure the backend is running and check for CORS issues if testing across different domains.

5. **Database Migration Errors**: If you encounter database migration errors, try deleting the `synapps.db` file and running `alembic upgrade head` again.

---

This document will be updated as development progresses and new issues or improvements are identified.
