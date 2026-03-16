# SynApps User Guide

Welcome to SynApps! This guide will help you get started with creating, running, and managing AI workflows.

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Dashboard](#dashboard)
4. [Creating Workflows](#creating-workflows)
5. [Running Workflows](#running-workflows)
6. [Monitoring & History](#monitoring--history)
7. [Applet Library](#applet-library)
8. [Customizing Applets](#customizing-applets)
9. [Settings](#settings)
10. [Troubleshooting](#troubleshooting)

## Introduction

SynApps is a visual platform for building modular AI workflows. Each workflow consists of applets (specialized AI agents) that work together to accomplish tasks. The platform allows you to create workflows using a visual editor, run them in the background, and monitor their progress in real-time.

Key concepts:
- **Applets**: Specialized AI agents that perform specific tasks (e.g., generating text, creating images)
- **Workflows**: Sequences of applets connected to work together
- **Orchestrator**: The system that routes messages between applets and manages workflow execution

## Getting Started

### First Login

When you first log in to SynApps, you'll be greeted with the Dashboard. From here, you can:
- Create a new workflow
- Use a pre-built template
- View your recent workflows and runs

### API Keys

To use certain applets, you'll need to set up API keys:
1. Navigate to the **Settings** page
2. Enter your API keys:
   - **OpenAI API Key**: Required for the Writer applet and AI code suggestions
   - **Stability AI API Key**: Required for the Artist applet

## Dashboard

The Dashboard provides an overview of your SynApps environment:

- **Templates**: Quick-start templates for common workflows
- **Recent Workflows**: Workflows you've created or edited recently
- **Recent Runs**: Recent workflow executions and their status

To create a new workflow from a template, simply click on a template card and choose "Create Flow".

## Creating Workflows

### Using Templates

The easiest way to create a workflow is to use a template:
1. From the Dashboard, click on a template
2. The template will be loaded into the Workflow Editor
3. Customize the workflow as needed
4. Click "Save" to save your changes

### Creating from Scratch

To create a workflow from scratch:
1. Navigate to the Editor page
2. Add nodes from the sidebar by dragging them onto the canvas
3. Connect nodes by dragging from one node's output handle to another node's input handle
4. Configure each node by clicking on it and adjusting settings
5. Click "Save" to save your workflow

### Node Types

SynApps supports several types of nodes:
- **Start**: The entry point for the workflow
- **End**: The exit point for the workflow
- **Writer**: Generates text using AI
- **Artist**: Creates images from text descriptions
- **Memory**: Stores and retrieves information between steps

## Running Workflows

To run a workflow:
1. Open the workflow in the Editor
2. Enter input data in the sidebar (as text or JSON)
3. Click "Run Workflow"

The workflow will execute in the background, and you'll see real-time updates on the canvas. Each node will show its status (running, success, error) as the workflow progresses.

### Background Execution

Once started, workflows run in the background. You can:
- Close the browser tab and return later
- Navigate to other parts of the application
- Start another workflow

You'll receive notifications when workflows complete or encounter errors.

## Monitoring & History

### Real-Time Monitoring

While a workflow is running, you can monitor its progress on the canvas:
- Nodes will change color based on their status
- Edges will animate to show data flow
- A progress bar will display overall completion

### Run History

To view completed runs:
1. Navigate to the History page
2. Select a run from the sidebar
3. View details including:
   - Start and end times
   - Status and duration
   - Results from each node
   - Any errors encountered

## Applet Library

The Applet Library shows all available applets and their capabilities:
1. Navigate to the Applet Library page
2. Browse available applets
3. Click on an applet to view details:
   - Description and capabilities
   - Usage examples
   - Version information

## Customizing Applets

Advanced users can customize applet behavior by editing their code:
1. In the Applet Library, select an applet
2. Click "View Code" to open the code editor
3. Make changes to the code
4. Use the AI Assistant for code suggestions
5. Click "Save Changes" to update the applet

### AI-Assisted Editing

The Code Editor includes an AI Assistant that can help you modify applet code:
1. Describe what you want to change in the input field
2. Click "Get Suggestion" to receive AI-generated code
3. Review the suggested changes
4. Click "Apply Changes" if you want to use the suggestion

## Settings

The Settings page allows you to configure:
- **API Keys**: For external services (OpenAI, Stability AI)
- **Appearance**: Visual settings for the interface
- **Notifications**: Browser and email notification preferences
- **User Interface**: Interface behavior options
- **Advanced**: Technical settings for developers

## Troubleshooting

### Common Issues

**Workflow doesn't start**
- Check that all required API keys are set
- Ensure the workflow has a Start node
- Verify all nodes are properly connected

**Node fails with an error**
- Check the error message in the run details
- Verify API keys for external services
- Adjust node settings if needed

**Results not as expected**
- Review the input data format
- Check node configurations
- Consider adjusting prompts or parameters

### Support

If you encounter issues not covered in this guide:
- Check the [documentation](https://docs.synapps.ai)
- Join the [Discord community](https://discord.gg/synapps)
- Submit a GitHub issue if you find a bug

## Advanced Usage

### Environment Variables

For advanced deployments, SynApps supports configuration via environment variables:
- `OPENAI_API_KEY`: OpenAI API key
- `STABILITY_API_KEY`: Stability AI API key
- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)
- `LOG_LEVEL`: Logging level (debug, info, warning, error)

### Development Mode

To run SynApps in development mode:
1. Set up the development environment as described in the README
2. Run the backend: `cd apps/orchestrator && uvicorn main:app --reload`
3. Run the frontend: `cd apps/web-frontend && npm start`
4. Access the application at http://localhost:3000
