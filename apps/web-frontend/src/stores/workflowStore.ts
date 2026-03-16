import { create } from "zustand";
import { Flow } from "@/types";
import { generateId } from "@/utils/flowUtils";

interface WorkflowStoreState {
  flow: Flow | null;
  isLoading: boolean;
  isSaving: boolean;
  showTemplates: boolean;
  showCodeEditor: boolean;
  selectedApplet: string;
  appletCode: string;
  inputData: string;
  imageGenerator: string;
  setFlow: (flow: Flow | null) => void;
  setIsLoading: (isLoading: boolean) => void;
  setIsSaving: (isSaving: boolean) => void;
  setShowTemplates: (showTemplates: boolean) => void;
  setShowCodeEditor: (showCodeEditor: boolean) => void;
  setSelectedApplet: (selectedApplet: string) => void;
  setAppletCode: (appletCode: string) => void;
  setInputData: (inputData: string) => void;
  setImageGenerator: (imageGenerator: string) => void;
  createEmptyFlow: () => Flow;
  resetWorkflowState: () => void;
}

const createDefaultFlow = (): Flow => ({
  id: generateId(),
  name: "New Workflow",
  nodes: [
    {
      id: generateId(),
      type: "start",
      position: { x: 250, y: 25 },
      data: { label: "Start" },
    },
    {
      id: generateId(),
      type: "end",
      position: { x: 250, y: 500 },
      data: { label: "End" },
    },
  ],
  edges: [],
});

export const useWorkflowStore = create<WorkflowStoreState>((set) => ({
  flow: null,
  isLoading: true,
  isSaving: false,
  showTemplates: false,
  showCodeEditor: false,
  selectedApplet: "",
  appletCode: "",
  inputData: "",
  imageGenerator: "stability",
  setFlow: (flow) => set({ flow }),
  setIsLoading: (isLoading) => set({ isLoading }),
  setIsSaving: (isSaving) => set({ isSaving }),
  setShowTemplates: (showTemplates) => set({ showTemplates }),
  setShowCodeEditor: (showCodeEditor) => set({ showCodeEditor }),
  setSelectedApplet: (selectedApplet) => set({ selectedApplet }),
  setAppletCode: (appletCode) => set({ appletCode }),
  setInputData: (inputData) => set({ inputData }),
  setImageGenerator: (imageGenerator) => set({ imageGenerator }),
  createEmptyFlow: () => {
    const newFlow = createDefaultFlow();
    set({ flow: newFlow });
    return newFlow;
  },
  resetWorkflowState: () =>
    set({
      flow: null,
      isLoading: true,
      isSaving: false,
      showTemplates: false,
      showCodeEditor: false,
      selectedApplet: "",
      appletCode: "",
      inputData: "",
      imageGenerator: "stability",
    }),
}));
