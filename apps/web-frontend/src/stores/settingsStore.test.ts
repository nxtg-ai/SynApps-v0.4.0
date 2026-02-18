import { vi } from 'vitest';
import { useSettingsStore } from './settingsStore';

describe('useSettingsStore', () => {
    beforeEach(() => {
        // Reset the store before each test
        useSettingsStore.setState({
            apiKey: '',
            setApiKey: vi.fn(),
        });
    });

    it('should set the API key', () => {
        const { setApiKey } = useSettingsStore.getState();
        const newApiKey = 'test-api-key';
        setApiKey(newApiKey);
        expect(useSettingsStore.getState().apiKey).toBe(newApiKey);
    });
});
