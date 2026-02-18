import { render, screen, fireEvent } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import TemplateLoader from './TemplateLoader';
import { Flow, FlowTemplate } from '@/types';
import { act } from '@testing-library/react';

// Use vi.hoisted to define mockTemplates so it's available when vi.mock is hoisted
const { mockTemplates } = vi.hoisted(() => {
  const mockTemplates: FlowTemplate[] = [
    {
      id: 'template-1',
      name: 'Template One',
      description: 'Description for template one',
      tags: ['tag1', 'tag2'],
      flow: { id: 'flow-1', name: 'Flow One', nodes: [], edges: [] },
    },
    {
      id: 'template-2',
      name: 'Template Two',
      description: 'Description for template two',
      tags: ['tag3'],
      flow: { id: 'flow-2', name: 'Flow Two', nodes: [], edges: [] },
    },
  ];
  return { mockTemplates };
});

// Mock the templates module using the hoisted mock
vi.mock('../../templates', () => ({
  templates: mockTemplates,
}));

// Mock createFlowFromTemplate
vi.mock('../../utils/flowUtils', () => ({
  createFlowFromTemplate: vi.fn((template: FlowTemplate) => ({
    ...template.flow,
    id: `new-${template.flow.id}`,
    name: `New ${template.flow.name}`,
  })),
}));

describe('TemplateLoader', () => {
  const onSelectTemplateMock = vi.fn();
  const onCloseMock = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  const renderTemplateLoader = () => {
    return render(
      <TemplateLoader onSelectTemplate={onSelectTemplateMock} onClose={onCloseMock} />
    );
  };

  it('should render all templates from the mock data', () => {
    renderTemplateLoader();
    expect(screen.getByText('Template One')).toBeInTheDocument();
    expect(screen.getByText('Description for template one')).toBeInTheDocument();
    expect(screen.getByText('Template Two')).toBeInTheDocument();
    expect(screen.getByText('Description for template two')).toBeInTheDocument();
  });

  it('should disable the "Create Flow" button initially', () => {
    renderTemplateLoader();
    const createButton = screen.getByRole('button', { name: /Create Flow/i });
    expect(createButton).toBeDisabled();
  });

  it('should enable the "Create Flow" button when a template is selected', () => {
    renderTemplateLoader();
    const createButton = screen.getByRole('button', { name: /Create Flow/i });
    expect(createButton).toBeDisabled();

    act(() => {
      fireEvent.click(screen.getByText('Template One'));
    });

    expect(createButton).not.toBeDisabled();
  });

  it('should call onSelectTemplate with a new flow when "Create Flow" is clicked', () => {
    renderTemplateLoader();
    const createButton = screen.getByRole('button', { name: /Create Flow/i });

    act(() => {
      fireEvent.click(screen.getByText('Template One'));
    });

    act(() => {
      fireEvent.click(createButton);
    });

    expect(onSelectTemplateMock).toHaveBeenCalledTimes(1);
    expect(onSelectTemplateMock).toHaveBeenCalledWith(
      expect.objectContaining({
        id: 'new-flow-1', // From mock createFlowFromTemplate
        name: 'New Flow One', // From mock createFlowFromTemplate
      })
    );
  });

  it('should call onClose when the close button is clicked', () => {
    renderTemplateLoader();
    const closeButton = screen.getByRole('button', { name: /×/i });

    act(() => {
      fireEvent.click(closeButton);
    });

    expect(onCloseMock).toHaveBeenCalledTimes(1);
  });

  it('should apply selected class to the clicked template card', () => {
    renderTemplateLoader();
    const templateOneCard = screen.getByText('Template One').closest('.template-card');
    const templateTwoCard = screen.getByText('Template Two').closest('.template-card');

    expect(templateOneCard).not.toHaveClass('selected');
    expect(templateTwoCard).not.toHaveClass('selected');

    act(() => {
      fireEvent.click(screen.getByText('Template One'));
    });

    expect(templateOneCard).toHaveClass('selected');
    expect(templateTwoCard).not.toHaveClass('selected');

    act(() => {
      fireEvent.click(screen.getByText('Template Two'));
    });

    expect(templateOneCard).not.toHaveClass('selected');
    expect(templateTwoCard).toHaveClass('selected');
  });
});