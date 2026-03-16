# SynApps Windsurf Workflow & Prompt Templates

This document provides comprehensive workflow templates and prompt structures for the SynApps project based on the principles in "The Complete Guide to Windsurf Project." These templates are designed to maximize development efficiency while maintaining code quality and adherence to established global rules.

## 1. Context Priming Template

This template should be used at the start of each development session to prime Windsurf with essential project context:

```markdown
# SynApps Project Context

## Project Overview
You are working on SynApps v0.4.0, a web-based visual platform for modular AI agents. The platform allows users to build autonomous AI applets (Writer, Memory, Artist, etc.) that work together through an orchestrator.

## Architecture
- **Backend**: Python 3.9+ with FastAPI microkernel, WebSockets, SQLAlchemy ORM, Alembic migrations
- **Frontend**: React 18, TypeScript, React Flow for workflow visualization
- **Database**: SQLite (dev), PostgreSQL (prod)
- **Applets**: Modular AI agents with standardized interfaces

## Current Development Phase
[Describe current phase: initial development/feature enhancement/bug fixing]

## Coding Standards
Follow the established Global Rules with emphasis on:
- Async programming with proper error handling in Python
- Functional components with hooks in React
- Strong typing with TypeScript
- Comprehensive testing (80%+ coverage)
- Accessible UI components (WCAG 2.1 AA)

## Current Task Context
[Describe the specific task or feature being worked on]
```

## 2. Feature Development Workflow

### Phase 1: Planning & Architecture

```markdown
# SynApps Feature Planning

## Role Definition
Act as a senior full-stack developer specializing in modular AI systems with expertise in Python/FastAPI backends, React/TypeScript frontends, and real-time communication via WebSockets.

## Feature Description
[Detailed description of the feature]

## Technical Requirements
- **Backend Changes**: [Specific APIs, database models, or services needed]
- **Frontend Changes**: [Components, state management, UI elements]
- **Data Flow**: [How data moves through the system]
- **Security Considerations**: [Auth requirements, input validation needs]

## Architecture Approach
I need you to design an architecture approach for this feature that:
1. Follows our microservices pattern
2. Maintains separation of concerns
3. Leverages existing applet interfaces
4. Ensures proper error handling and state management
5. Considers performance implications

Please provide an architectural diagram using text/ASCII and a component breakdown before proceeding to implementation.
```

### Phase 2: Backend Implementation

```markdown
# SynApps Backend Implementation

## Role Definition
Act as a senior Python backend developer specializing in FastAPI, SQLAlchemy, WebSockets, and asynchronous programming patterns.

## Feature Context
[Brief feature description and reference to architecture plan]

## Implementation Requirements
1. Implement the following APIs/services:
   - [List specific endpoints with methods, parameters, and responses]

2. Create/modify these database models:
   - [List models with fields and relationships]

3. Ensure proper validation using Pydantic v2 models
4. Implement comprehensive error handling and logging
5. Add unit tests with pytest achieving 80%+ coverage
6. Follow our established Python coding standards

## Quality Gates
- Passes all tests
- Implements proper validation for all inputs
- Uses async/await consistently
- Includes proper error handling
- Follows established project patterns
```

### Phase 3: Frontend Implementation

```markdown
# SynApps Frontend Implementation

## Role Definition
Act as a senior React developer specializing in TypeScript, React Flow, WebSocket integration, and modern React patterns.

## Feature Context
[Brief feature description and reference to architecture plan]

## Implementation Requirements
1. Create/modify the following components:
   - [List components with props and behaviors]

2. Implement state management using:
   - [Specify approach - Zustand, Context API, etc.]

3. Add these UI elements:
   - [List specific UI elements]

4. Implement WebSocket connections for real-time updates
5. Add unit tests with React Testing Library
6. Ensure accessibility compliance (WCAG 2.1 AA)

## Quality Gates
- Responsive design works on all target devices
- Meets accessibility standards
- All props properly typed with TypeScript
- Follows component structure in global rules
- Properly handles loading/error states
```

## 3. Applet Development Template

```markdown
# SynApps Applet Development

## Role Definition
Act as a senior AI systems developer specializing in modular agent development with Python, API integrations, and state management.

## Applet Description
[Name and purpose of the applet]

## Technical Requirements
1. Extend the BaseApplet class
2. Implement these primary methods:
   - [List required methods]

3. Handle these external API integrations:
   - [List APIs with authentication patterns]

4. Implement error handling and retry logic
5. Add unit tests covering both success and failure paths
6. Document usage examples and configuration options

## Quality Gates
- Gracefully handles API failures
- Properly manages API keys via environment variables
- Includes comprehensive logging
- All error paths properly handled
- Documentation complete with examples
```

## 4. Bug Fix Workflow

```markdown
# SynApps Bug Fix

## Role Definition
Act as a debugging specialist with expertise in Python/FastAPI and React/TypeScript applications.

## Bug Description
[Detailed description of the bug with steps to reproduce]

## Error Information
- **Error Message**: [Exact error message]
- **Stack Trace**: [Relevant parts of stack trace]
- **Affected Components**: [List of components/services involved]

## Debugging Approach
I need you to:
1. Analyze the root cause of this issue
2. Propose a solution that addresses the core problem
3. Implement the fix with minimal changes to existing architecture
4. Add or modify tests to ensure this issue doesn't recur
5. Document the fix and any potential side effects

## Quality Gates
- Fix does not introduce new issues
- Root cause is fully addressed
- Tests added/modified to catch this issue
- Documentation updated
```

## 5. Code Review Template

```markdown
# SynApps Code Review

## Role Definition
Act as a senior code reviewer with expertise in Python/FastAPI and React/TypeScript best practices.

## Review Context
I need you to review the following code for:
1. Adherence to our Global Rules
2. Potential security vulnerabilities
3. Performance optimization opportunities
4. Error handling completeness
5. Testing coverage
6. Architectural consistency

## Code to Review
[Insert code snippets or file paths]

## Review Format
Please provide:
1. A summary of overall code quality
2. Specific issues categorized by severity
3. Recommended improvements
4. Examples of proper implementations where needed
```

## 6. Implementation Directory Structure

To implement these workflows in your project, create the following directory structure:

```
.windsurf/
├── workflows/
│   ├── context-priming.json
│   ├── feature-development.json
│   ├── applet-development.json
│   ├── bug-fix.json
│   └── code-review.json
├── rules/
│   ├── python-standards.json
│   ├── react-standards.json
│   ├── security-standards.json
│   └── testing-standards.json
└── templates/
    ├── backend-api.py
    ├── react-component.tsx
    ├── applet-base.py
    └── test-templates.py
```

## 7. Activation Strategy

Configure workflow activation using these triggers:

1. **Always On**:
   - Python/React coding standards
   - Security validation rules
   - Documentation requirements

2. **Pattern-Based**:
   - Applet development workflow when editing files in `apps/applets/`
   - Backend API workflows when editing files in `apps/orchestrator/`
   - Frontend component workflows when editing `.tsx` files

3. **Explicit Triggers**:
   - Feature development with keyword "implement feature"
   - Bug fixing with keyword "fix bug"
   - Code review with keyword "review code"
