# Contributing to SynApps v0.4.0

Thank you for your interest in contributing to SynApps! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
  - [Prerequisites](#prerequisites)
  - [Local Setup](#local-setup)
  - [Running the Application](#running-the-application)
- [Coding Standards](#coding-standards)
  - [Python](#python)
  - [TypeScript/React](#typescriptreact)
- [Pull Request Process](#pull-request-process)
- [Creating Applets](#creating-applets)
- [Testing](#testing)
  - [Backend Testing](#backend-testing)
  - [Frontend Testing](#frontend-testing)
  - [End-to-End Testing](#end-to-end-testing)
- [Documentation](#documentation)
- [Deployment](#deployment)
- [Community](#community)

## Code of Conduct

Our project adheres to a Code of Conduct that establishes expected behavior in our community. Please read [the full text](CODE_OF_CONDUCT.md) to understand what actions will and will not be tolerated.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork to your local machine:
   ```
   git clone https://github.com/yourusername/synapps-mvp.git
   ```
3. Add the original repository as an upstream remote:
   ```
   git remote add upstream https://github.com/synapps/synapps-mvp.git
   ```
4. Create a new branch for your feature or bugfix:
   ```
   git checkout -b feature/your-feature-name
   ```

## Development Environment

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) (for containerized development)
- [Node.js](https://nodejs.org/) 16+ and npm (for frontend development)
- [Python](https://www.python.org/downloads/) 3.9+ and pip (for backend development)
- API keys for [OpenAI](https://platform.openai.com/) and [Stability AI](https://stability.ai/) (for testing AI features)
- Git (for version control)

### Local Setup

1. Fork the repository on GitHub and clone your fork:
   ```bash
   git clone https://github.com/yourusername/SynApps-v0.4.0.git
   cd SynApps-v0.4.0
   ```

2. Create a `.env.development` file in the root directory with your API keys and database configuration:
   ```
   OPENAI_API_KEY=your_openai_api_key
   STABILITY_API_KEY=your_stability_api_key
   DATABASE_URL=sqlite+aiosqlite:///synapps.db
   BACKEND_CORS_ORIGINS=http://localhost:3000
   ```

3. Set up the backend environment:
   ```bash
   cd apps/orchestrator
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .  # Installs the package in editable mode with all dependencies
   ```

4. Initialize the database:
   ```bash
   alembic upgrade head
   ```

5. Install frontend dependencies:
   ```bash
   cd ../web-frontend
   npm install
   ```

### Running the Application

#### Using Docker

To run the entire application using Docker:

```bash
docker-compose -f infra/docker/docker-compose.yml up --build
```

Then access the application at http://localhost:3000

#### Manual Development

For backend development, in one terminal:
```bash
cd apps/orchestrator
source venv/bin/activate  # On Windows: venv\Scripts\activate
uvicorn main:app --reload --port 8000
```

For frontend development, in another terminal:
```bash
cd apps/web-frontend
npm start
```

The frontend will be available at http://localhost:3000 and the backend API at http://localhost:8000

## Coding Standards

### Python

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for code style
- Use type hints where possible
- Document functions and classes with docstrings
- Use async/await for I/O-bound operations

### TypeScript/React

- Follow the [Airbnb JavaScript Style Guide](https://github.com/airbnb/javascript)
- Use functional components with hooks
- Use TypeScript interfaces for props and state
- Keep components focused on a single responsibility

## Pull Request Process

1. Ensure your code follows the coding standards
2. Update documentation if necessary
3. Add tests for new functionality
4. Make sure all tests pass
5. Submit a pull request to the `main` branch
6. Wait for a maintainer to review your PR
7. Address any feedback from reviewers
8. Once approved, a maintainer will merge your PR

## Creating Applets

SynApps is designed to be extensible through custom applets. To create a new applet:

1. Create a new directory in `apps/applets/` with your applet name
2. Implement a class that extends `BaseApplet` and implements the required methods
3. Add tests for your applet in the `tests` directory
4. Document your applet's capabilities and usage
5. Submit a pull request following the process above

Example applet structure:

```
apps/applets/my-applet/
├── applet.py      # Main applet implementation
├── setup.py       # Package setup file
└── tests/         # Tests for your applet
    └── test_applet.py
```

## Testing

### Backend Testing

We use pytest for backend testing. The tests are located in `apps/orchestrator/tests/`.

To run all backend tests:

```bash
cd apps/orchestrator
source venv/bin/activate  # On Windows: venv\Scripts\activate
pytest -v
```

To run a specific test file:

```bash
pytest tests/test_main.py -v
```

To run with coverage reporting:

```bash
pytest --cov=. --cov-report=term-missing
```

When writing backend tests:
- Use the `TestClient` from FastAPI for API testing
- Mock external dependencies when appropriate
- Test both success and error cases
- Ensure database operations are properly isolated

### Frontend Testing

We use Jest and React Testing Library for frontend testing. The tests are located alongside the components they test with a `.test.tsx` extension.

To run all frontend tests:

```bash
cd apps/web-frontend
npm test
```

To run tests in watch mode (useful during development):

```bash
npm test -- --watch
```

To run a specific test file:

```bash
npm test -- src/components/WorkflowCanvas/nodes/AppletNode.test.tsx
```

To run with coverage reporting:

```bash
npm test -- --coverage
```

When writing frontend tests:
- Focus on testing component behavior, not implementation details
- Mock API calls and external dependencies
- Test user interactions using `userEvent` from Testing Library
- Verify that components render correctly with different props

### End-to-End Testing

E2E tests verify that the entire application works correctly from the user's perspective.

For manual E2E testing, please verify these critical workflows:
1. Creating and saving a new workflow
2. Running a workflow with different inputs
3. Viewing workflow run history
4. Editing an existing workflow

When contributing, please ensure that your changes don't break these critical user flows.

## Documentation

Good documentation is crucial for the project. Please update the following as needed:

- README.md for overview and quick start
- In-line code comments for complex logic
- API documentation using docstrings
- Update architecture.md for design changes
- Add usage examples for new features

When documenting code:

- For Python, use Google-style docstrings
- For TypeScript/React, use JSDoc comments
- Include examples where appropriate
- Document both public APIs and internal functions with complex logic
- Keep documentation up to date when changing code

## Deployment

SynApps can be deployed in several ways:

### Docker Deployment

For production deployment using Docker:

```bash
docker-compose -f infra/docker/docker-compose.yml -f infra/docker/docker-compose.prod.yml up -d
```

### Kubernetes Deployment

For Kubernetes deployment, refer to the documentation in `infra/k8s/README.md`.

## Community

Join our community to discuss development, get help, and share your work:

- [GitHub Discussions](https://github.com/nxtg-ai/SynApps-v0.4.0/discussions)
- [GitHub Issues](https://github.com/nxtg-ai/SynApps-v0.4.0/issues)

## License

By contributing to SynApps, you agree that your contributions will be licensed under the project's MIT License.
