import { create } from "zustand";
import { WorkflowRunStatus } from "@/types";

interface ExecutionStoreState {
  isRunning: boolean;
  workflowResults: Record<string, any> | null;
  runStatus: WorkflowRunStatus | null;
  completedNodes: string[];
  setIsRunning: (isRunning: boolean) => void;
  setWorkflowResults: (workflowResults: Record<string, any> | null) => void;
  setRunStatus: (runStatus: WorkflowRunStatus | null) => void;
  setCompletedNodes: (completedNodes: string[]) => void;
  resetExecutionState: () => void;
}

export const useExecutionStore = create<ExecutionStoreState>((set) => ({
  isRunning: false,
  workflowResults: null,
  runStatus: null,
  completedNodes: [],
  setIsRunning: (isRunning) => set({ isRunning }),
  setWorkflowResults: (workflowResults) => set({ workflowResults }),
  setRunStatus: (runStatus) => set({ runStatus }),
  setCompletedNodes: (completedNodes) => set({ completedNodes }),
  resetExecutionState: () =>
    set({
      isRunning: false,
      workflowResults: null,
      runStatus: null,
      completedNodes: [],
    }),
}));
