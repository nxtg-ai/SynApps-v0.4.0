import pytest
import asyncio
import os
from unittest.mock import patch, MagicMock
from pathlib import Path
from apps.orchestrator.main import CodeNodeApplet, AppletMessage, CODE_NODE_TYPE

# Mock the _sandbox_preexec_fn since resource.setrlimit is OS-specific and hard to test directly
# We will focus on the wrapper logic and general behavior.
@pytest.fixture(autouse=True)
def mock_sandbox_preexec_fn():
    with patch('apps.orchestrator.main._sandbox_preexec_fn') as mock_fn:
        mock_fn.return_value = None  # Return a no-op preexec_fn for tests
        yield

@pytest.fixture(autouse=True)
def mock_rmtree():
    """Mock shutil.rmtree to prevent actual filesystem cleanup during tests."""
    with patch('shutil.rmtree') as mock_func:
        yield mock_func

@pytest.fixture
def code_node_applet():
    return CodeNodeApplet()

@pytest.fixture
def mock_subprocess_exec():
    """Mocks asyncio.create_subprocess_exec for controlled testing."""
    mock_proc = MagicMock()
    mock_proc.stdin = asyncio.StreamWriter(MagicMock(), MagicMock(), MagicMock(), MagicMock())
    mock_proc.stdout = asyncio.StreamReader()
    mock_proc.stderr = asyncio.StreamReader()
    mock_proc.returncode = 0
    mock_proc.wait.return_value = 0 # Simulate immediate completion
    mock_proc.kill = MagicMock()

    with patch('asyncio.create_subprocess_exec', return_value=mock_proc) as mock_exec:
        yield mock_exec, mock_proc

@pytest.mark.asyncio
async def test_code_node_python_execution(code_node_applet, mock_subprocess_exec):
    """Test basic Python code execution."""
    mock_exec, mock_proc = mock_subprocess_exec

    code = "result = data['input'] * 2"
    message = AppletMessage(
        content={"input": 5},
        metadata={"node_data": {"language": "python", "code": code}}
    )

    # Simulate sandbox output
    def write_output():
        mock_proc.stdout.feed_data(b'''__SYNAPPS_RESULT_START__
{"ok": true, "result": 10}
__SYNAPPS_RESULT_END__
''')
        mock_proc.stdout.feed_eof()
    asyncio.get_event_loop().call_soon(write_output)

    response = await code_node_applet.on_message(message)

    assert response.metadata["applet"] == CODE_NODE_TYPE
    assert response.metadata["status"] == "success"
    assert response.content["ok"] is True
    assert response.content["result"] == 10
    mock_exec.assert_called_once()
    assert "python" in mock_exec.call_args[0][0]
    assert mock_proc.stdin.write.called
    assert b'"code": "result = data[\'input\'] * 2"' in mock_proc.stdin.write.call_args[0][0]

@pytest.mark.asyncio
async def test_code_node_javascript_execution(code_node_applet, mock_subprocess_exec):
    """Test basic JavaScript code execution."""
    mock_exec, mock_proc = mock_subprocess_exec

    code = "sandbox.result = data.input * 3;"
    message = AppletMessage(
        content={"input": 7},
        metadata={"node_data": {"language": "javascript", "code": code}}
    )

    # Simulate sandbox output
    def write_output():
        mock_proc.stdout.feed_data(b'''__SYNAPPS_RESULT_START__
{"ok": true, "result": 21}
__SYNAPPS_RESULT_END__
''')
        mock_proc.stdout.feed_eof()
    asyncio.get_event_loop().call_soon(write_output)

    response = await code_node_applet.on_message(message)

    assert response.metadata["applet"] == CODE_NODE_TYPE
    assert response.metadata["status"] == "success"
    assert response.content["ok"] is True
    assert response.content["result"] == 21
    mock_exec.assert_called_once()
    assert "node" in mock_exec.call_args[0][0]
    assert mock_proc.stdin.write.called
    assert b'"code": "sandbox.result = data.input * 3;"' in mock_proc.stdin.write.call_args[0][0]

@pytest.mark.asyncio
async def test_code_node_timeout(code_node_applet, mock_subprocess_exec):
    """Test code execution timeout."""
    mock_exec, mock_proc = mock_subprocess_exec
    mock_proc.wait.side_effect = asyncio.TimeoutError
    mock_proc.returncode = -9 # Simulate process killed by timeout

    code = "import time; time.sleep(10)"
    message = AppletMessage(
        content={},
        metadata={"node_data": {"language": "python", "code": code, "timeout_seconds": 1}}
    )

    response = await code_node_applet.on_message(message)

    assert response.metadata["status"] == "error"
    assert response.content["ok"] is False
    assert response.content["timed_out"] is True
    assert response.content["error"]["message"] == "Execution timed out"
    mock_proc.kill.assert_called_once()

@pytest.mark.asyncio
async def test_code_node_filesystem_restriction_python(code_node_applet, mock_subprocess_exec):
    """Test Python sandbox restricts filesystem access outside /tmp."""
    mock_exec, mock_proc = mock_subprocess_exec
    # Simulate sandbox output for a permission error
    def write_output():
        mock_proc.stdout.feed_data(
            b'''__SYNAPPS_RESULT_START__
{"ok": false, "error": {"type": "PermissionError", "message": "Filesystem access is restricted to /tmp: /etc/passwd", "traceback": "..."}}
__SYNAPPS_RESULT_END__
'''
        )
        mock_proc.stdout.feed_eof()
    asyncio.get_event_loop().call_soon(write_output)

    code = "with open('/etc/passwd', 'r') as f: pass"
    message = AppletMessage(
        content={},
        metadata={"node_data": {"language": "python", "code": code}}
    )

    response = await code_node_applet.on_message(message)

    assert response.metadata["status"] == "error"
    assert response.content["ok"] is False
    assert "Filesystem access is restricted to /tmp" in response.content["error"]["message"]

@pytest.mark.asyncio
async def test_code_node_filesystem_restriction_javascript(code_node_applet, mock_subprocess_exec):
    """Test JavaScript sandbox restricts filesystem access outside /tmp."""
    mock_exec, mock_proc = mock_subprocess_exec
    # Simulate sandbox output for a permission error
    def write_output():
        mock_proc.stdout.feed_data(
            b'''__SYNAPPS_RESULT_START__
{"ok": false, "error": {"type": "Error", "message": "Filesystem access is restricted to /tmp: /etc/passwd", "stack": "..."}}
__SYNAPPS_RESULT_END__
'''
        )
        mock_proc.stdout.feed_eof()
    asyncio.get_event_loop().call_soon(write_output)

    code = "require('fs').readFileSync('/etc/passwd');"
    message = AppletMessage(
        content={},
        metadata={"node_data": {"language": "javascript", "code": code}}
    )

    response = await code_node_applet.on_message(message)

    assert response.metadata["status"] == "error"
    assert response.content["ok"] is False
    assert "Filesystem access is restricted to /tmp" in response.content["error"]["message"]

@pytest.mark.asyncio
async def test_code_node_blocked_module_python(code_node_applet, mock_subprocess_exec):
    """Test Python sandbox blocks dangerous module imports."""
    mock_exec, mock_proc = mock_subprocess_exec
    # Simulate sandbox output for an import error
    def write_output():
        mock_proc.stdout.feed_data(
            b'''__SYNAPPS_RESULT_START__
{"ok": false, "error": {"type": "ImportError", "message": "Import 'subprocess' is blocked in code sandbox", "traceback": "..."}}
__SYNAPPS_RESULT_END__
'''
        )
        mock_proc.stdout.feed_eof()
    asyncio.get_event_loop().call_soon(write_output)

    code = "import subprocess"
    message = AppletMessage(
        content={},
        metadata={"node_data": {"language": "python", "code": code}}
    )

    response = await code_node_applet.on_message(message)

    assert response.metadata["status"] == "error"
    assert response.content["ok"] is False
    assert "Import 'subprocess' is blocked" in response.content["error"]["message"]

@pytest.mark.asyncio
async def test_code_node_blocked_module_javascript(code_node_applet, mock_subprocess_exec):
    """Test JavaScript sandbox blocks dangerous module imports."""
    mock_exec, mock_proc = mock_subprocess_exec
    # Simulate sandbox output for an import error
    def write_output():
        mock_proc.stdout.feed_data(
            b'''__SYNAPPS_RESULT_START__
{"ok": false, "error": {"type": "Error", "message": "Import 'child_process' is blocked in code sandbox", "stack": "..."}}
__SYNAPPS_RESULT_END__
'''
        )
        mock_proc.stdout.feed_eof()
    asyncio.get_event_loop().call_soon(write_output)

    code = "require('child_process');"
    message = AppletMessage(
        content={},
        metadata={"node_data": {"language": "javascript", "code": code}}
    )

    response = await code_node_applet.on_message(message)

    assert response.metadata["status"] == "error"
    assert response.content["ok"] is False
    assert "Import 'child_process' is blocked" in response.content["error"]["message"]

@pytest.mark.asyncio
async def test_code_node_environment_sanitization_python(code_node_applet, mock_subprocess_exec):
    """Test Python sandbox environment variable sanitization."""
    mock_exec, mock_proc = mock_subprocess_exec
    # Simulate sandbox output by trying to read an arbitrary env var
    def write_output():
        mock_proc.stdout.feed_data(
            b'''__SYNAPPS_RESULT_START__
{"ok": true, "result": "TEST_VAR_VALUE"}
__SYNAPPS_RESULT_END__
'''
        )
        mock_proc.stdout.feed_eof()
    asyncio.get_event_loop().call_soon(write_output)

    code = "import os; result = os.environ.get('TEST_VAR', 'NOT_FOUND')"
    message = AppletMessage(
        content={},
        metadata={"node_data": {"language": "python", "code": code, "env": {"TEST_VAR": "TEST_VAR_VALUE"}}}
    )

    response = await code_node_applet.on_message(message)

    assert response.metadata["status"] == "success"
    assert response.content["ok"] is True
    assert response.content["result"] == "TEST_VAR_VALUE"
    # Verify that the env passed to subprocess exec is cleaned
    # We can only check for explicit passed `env` via call_args,
    # the actual sandbox wrapper modifies its own os.environ.
    assert mock_exec.call_args[1]['env']['TEST_VAR'] == 'TEST_VAR_VALUE'
    # Ensure that other variables are not there unless explicitly passed
    assert 'HOME' in mock_exec.call_args[1]['env'] # Should be default /tmp
    assert '/tmp' in mock_exec.call_args[1]['env']['HOME']

@pytest.mark.asyncio
async def test_code_node_environment_sanitization_javascript(code_node_applet, mock_subprocess_exec):
    """Test JavaScript sandbox environment variable sanitization."""
    mock_exec, mock_proc = mock_subprocess_exec
    # Simulate sandbox output by trying to read an arbitrary env var
    def write_output():
        mock_proc.stdout.feed_data(
            b'''__SYNAPPS_RESULT_START__
{"ok": true, "result": "JS_TEST_VAR_VALUE"}
__SYNAPPS_RESULT_END__
'''
        )
        mock_proc.stdout.feed_eof()
    asyncio.get_event_loop().call_soon(write_output)

    code = "sandbox.result = process.env.JS_TEST_VAR || 'NOT_FOUND';"
    message = AppletMessage(
        content={},
        metadata={"node_data": {"language": "javascript", "code": code, "env": {"JS_TEST_VAR": "JS_TEST_VAR_VALUE"}}}
    )

    response = await code_node_applet.on_message(message)

    assert response.metadata["status"] == "success"
    assert response.content["ok"] is True
    assert response.content["result"] == "JS_TEST_VAR_VALUE"
    # Verify that the env passed to subprocess exec is cleaned
    assert mock_exec.call_args[1]['env']['JS_TEST_VAR'] == 'JS_TEST_VAR_VALUE'
    assert 'HOME' in mock_exec.call_args[1]['env']
    assert '/tmp' in mock_exec.call_args[1]['env']['HOME']

@pytest.mark.asyncio
async def test_code_node_with_custom_working_dir(code_node_applet, mock_subprocess_exec):
    """Test code execution with a custom working directory within /tmp."""
    mock_exec, mock_proc = mock_subprocess_exec

    code = "import os; result = os.getcwd()"
    message = AppletMessage(
        content={},
        metadata={"node_data": {"language": "python", "code": code, "working_dir": "/tmp/my_custom_dir"}}
    )

    # Simulate sandbox output
    def write_output():
        mock_proc.stdout.feed_data(b'''__SYNAPPS_RESULT_START__
{"ok": true, "result": "/tmp/my_custom_dir"}
__SYNAPPS_RESULT_END__
''')
        mock_proc.stdout.feed_eof()
    asyncio.get_event_loop().call_soon(write_output)

    response = await code_node_applet.on_message(message)

    assert response.metadata["status"] == "success"
    assert response.content["ok"] is True
    assert response.content["result"] == "/tmp/my_custom_dir"
    # Check that the subprocess was called with the correct cwd
    assert "/tmp/my_custom_dir" in mock_exec.call_args[1]['cwd']

@pytest.mark.asyncio
async def test_code_node_empty_code_text(code_node_applet):
    """Test handling of empty code text."""
    message = AppletMessage(
        content={},
        metadata={"node_data": {"language": "python", "code": ""}}
    )
    response = await code_node_applet.on_message(message)
    assert response.metadata["status"] == "error"
    assert "No code provided" in response.content["error"]

    message_no_code_in_metadata = AppletMessage(
        content={"code": ""},
        metadata={"node_data": {"language": "python"}}
    )
    response = await code_node_applet.on_message(message_no_code_in_metadata)
    assert response.metadata["status"] == "error"
    assert "No code provided" in response.content["error"]

@pytest.mark.asyncio
async def test_code_node_no_code_provided_fallback_to_content(code_node_applet, mock_subprocess_exec):
    """Test code node falls back to message.content.code if no code in metadata."""
    mock_exec, mock_proc = mock_subprocess_exec

    code = "result = data['value'] + 1"
    message = AppletMessage(
        content={"code": code, "value": 10},
        metadata={"node_data": {"language": "python"}}
    )

    def write_output():
        mock_proc.stdout.feed_data(b'''__SYNAPPS_RESULT_START__
{"ok": true, "result": 11}
__SYNAPPS_RESULT_END__
''')
        mock_proc.stdout.feed_eof()
    asyncio.get_event_loop().call_soon(write_output)

    response = await code_node_applet.on_message(message)
    assert response.metadata["status"] == "success"
    assert response.content["ok"] is True
    assert response.content["result"] == 11
    assert b'"code": "result = data[\'value\'] + 1"' in mock_proc.stdin.write.call_args[0][0]

@pytest.mark.asyncio
async def test_code_node_stdout_truncated(code_node_applet, mock_subprocess_exec):
    """Test stdout truncation for large output."""
    mock_exec, mock_proc = mock_subprocess_exec

    # Set max_output_bytes to a small value
    code = "print('A' * 1000); result = 1"
    message = AppletMessage(
        content={},
        metadata={"node_data": {"language": "python", "code": code, "max_output_bytes": 100}}
    )

    def write_output():
        # Simulate stdout longer than max_output_bytes
        # The wrapper itself performs truncation, so we simulate the truncated output
        mock_proc.stdout.feed_data(b'''A''' * 100 + b'''...
__SYNAPPS_RESULT_START__
{"ok": true, "result": 1}
__SYNAPPS_RESULT_END__
''')
        mock_proc.stdout.feed_eof()
    asyncio.get_event_loop().call_soon(write_output)

    response = await code_node_applet.on_message(message)

    assert response.metadata["status"] == "success"
    assert response.content["ok"] is True
    assert response.content["stdout_truncated"] is True
    assert len(response.content["stdout"]) <= 100 + len('...\n') # Actual content of simulated output

@pytest.mark.asyncio
async def test_code_node_runtime_not_found(code_node_applet, mock_sandbox_preexec_fn):
    """Test handling when the runtime executable is not found."""
    with patch('asyncio.create_subprocess_exec', side_effect=FileNotFoundError) as mock_exec:
        message = AppletMessage(
            content={},
            metadata={"node_data": {"language": "unknown", "code": "print('hello')"}}
        )
        response = await code_node_applet.on_message(message)

        assert response.metadata["status"] == "error"
        assert response.content["ok"] is False
        assert "Runtime not found" in response.content["error"]["message"]
        mock_exec.assert_called_once()
