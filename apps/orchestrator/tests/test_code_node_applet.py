import pytest
import asyncio
import os
from unittest.mock import patch, MagicMock, AsyncMock
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
    # Mock stdin/stdout/stderr as MagicMock objects
    mock_proc.stdin = MagicMock()
    mock_proc.stdout = AsyncMock()
    mock_proc.stderr = AsyncMock()

    # Configure stdin methods
    mock_proc.stdin.write = MagicMock()
    mock_proc.stdin.drain = AsyncMock()
    mock_proc.stdin.close = MagicMock()

    # Default side_effect for stdout/stderr read to simulate stream ending
    mock_proc.stdout.read.side_effect = [b'', b''] # Default to empty, tests will set actual content
    mock_proc.stderr.read.side_effect = [b'', b''] # Default to empty

    mock_proc.returncode = 0
    mock_proc.wait = AsyncMock(return_value=0) # Simulate immediate completion
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
    mock_proc.stdout.read.side_effect = [b'''__SYNAPPS_RESULT_START__
{"ok": true, "result": 10}
__SYNAPPS_RESULT_END__
''', b'']

    response = await code_node_applet.on_message(message)

    assert response.metadata["applet"] == CODE_NODE_TYPE
    assert response.metadata["status"] == "success"
    assert response.content["ok"] is True
    assert response.content["result"] == 10
    mock_exec.assert_called_once()
    assert "python" in mock_exec.call_args[0][0]
    mock_proc.stdin.write.assert_called_once()
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
    mock_proc.stdout.read.side_effect = [b'''__SYNAPPS_RESULT_START__
{"ok": true, "result": 21}
__SYNAPPS_RESULT_END__
''', b'']

    response = await code_node_applet.on_message(message)

    assert response.metadata["applet"] == CODE_NODE_TYPE
    assert response.metadata["status"] == "success"
    assert response.content["ok"] is True
    assert response.content["result"] == 21
    mock_exec.assert_called_once()
    assert "node" in mock_exec.call_args[0][0]
    mock_proc.stdin.write.assert_called_once()
    assert b'"code": "sandbox.result = data.input * 3;"' in mock_proc.stdin.write.call_args[0][0]

@pytest.mark.asyncio
async def test_code_node_timeout(code_node_applet, mock_subprocess_exec):
    """Test code execution timeout."""
    mock_exec, mock_proc = mock_subprocess_exec
    # Patch asyncio.wait_for directly to raise TimeoutError
    with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError):
        # Ensure stdout/stderr are exhausted immediately for timeout test
        mock_proc.stdout.read.side_effect = [b'', b'']
        mock_proc.stderr.read.side_effect = [b'', b'']

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
    mock_proc.stdout.read.side_effect = [
        b'''__SYNAPPS_RESULT_START__
{"ok": false, "error": {"type": "PermissionError", "message": "Filesystem access is restricted to /tmp: /etc/passwd", "traceback": "..."}}
__SYNAPPS_RESULT_END__
''',
        b''
    ]

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
    mock_proc.stdout.read.side_effect = [
        b'''__SYNAPPS_RESULT_START__
{"ok": false, "error": {"type": "Error", "message": "Filesystem access is restricted to /tmp: /etc/passwd", "stack": "..."}}
__SYNAPPS_RESULT_END__
''',
        b''
    ]

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
    mock_proc.stdout.read.side_effect = [
        b'''__SYNAPPS_RESULT_START__
{"ok": false, "error": {"type": "ImportError", "message": "Import 'subprocess' is blocked in code sandbox", "traceback": "..."}}
__SYNAPPS_RESULT_END__
''',
        b''
    ]

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
    mock_proc.stdout.read.side_effect = [
        b'''__SYNAPPS_RESULT_START__
{"ok": false, "error": {"type": "Error", "message": "Import 'child_process' is blocked in code sandbox", "stack": "..."}}
__SYNAPPS_RESULT_END__
''',
        b''
    ]

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
    mock_proc.stdout.read.side_effect = [
        b'''__SYNAPPS_RESULT_START__
{"ok": true, "result": "TEST_VAR_VALUE"}
__SYNAPPS_RESULT_END__
''',
        b''
    ]

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
    assert mock_exec.call_args[1]['env']['TEST_VAR'] == 'TEST_VAR_VALUE' # Fixed typo here
    assert 'HOME' in mock_exec.call_args[1]['env'] # Should be default /tmp
    assert '/tmp' in mock_exec.call_args[1]['env']['HOME']

@pytest.mark.asyncio
async def test_code_node_environment_sanitization_javascript(code_node_applet, mock_subprocess_exec):
    """Test JavaScript sandbox environment variable sanitization."""
    mock_exec, mock_proc = mock_subprocess_exec
    # Simulate sandbox output by trying to read an arbitrary env var
    mock_proc.stdout.read.side_effect = [
        b'''__SYNAPPS_RESULT_START__
{"ok": true, "result": "JS_TEST_VAR_VALUE"}
__SYNAPPS_RESULT_END__
''',
        b''
    ]

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
    mock_proc.stdout.read.side_effect = [b'''__SYNAPPS_RESULT_START__
{"ok": true, "result": "/tmp/my_custom_dir"}
__SYNAPPS_RESULT_END__
''', b'']

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

    mock_proc.stdout.read.side_effect = [b'''__SYNAPPS_RESULT_START__
{"ok": true, "result": 11}
__SYNAPPS_RESULT_END__
''', b'']

    response = await code_node_applet.on_message(message)
    assert response.metadata["status"] == "success"
    assert response.content["ok"] is True
    assert response.content["result"] == 11
    assert b'"code": "result = data[\'value\'] + 1"' in mock_proc.stdin.write.call_args[0][0]

@pytest.mark.asyncio
async def test_code_node_stdout_truncated(code_node_applet): # Removed mock_subprocess_exec
    """Test stdout truncation for large output."""
    # We mock _read_stream_limited directly to ensure we control its return values
    # and properly test the handling of the 'truncated' flag
    with patch('apps.orchestrator.main._read_stream_limited', new_callable=AsyncMock) as mock_read_stream_limited:
        # It's important to mock _read_stream_limited within the test, not globally,
        # so it doesn't interfere with other tests that rely on real stream reading logic
        # if mock_subprocess_exec sets up a mock process.

        max_output_configured = 1024 # Changed to >= 1024 to pass validation
        code = "print('A' * 1000); result = 1"
        message = AppletMessage(
            content={},
            metadata={"node_data": {"language": "python", "code": code, "max_output_bytes": max_output_configured}}
        )

        json_result_part = b'''__SYNAPPS_RESULT_START__
{"ok": true, "result": 1}
__SYNAPPS_RESULT_END__
'''
        # The bytes that _read_stream_limited will *actually return* to _execute_sandboxed_code
        # This string *must* contain the result markers.
        simulated_stdout_bytes = json_result_part
        simulated_stderr_bytes = b''

        mock_read_stream_limited.side_effect = [
            (simulated_stdout_bytes, True),  # For stdout: simulate content and truncated flag
            (simulated_stderr_bytes, False)  # For stderr: simulate empty and not truncated
        ]

        # Patch asyncio.create_subprocess_exec to ensure _execute_sandboxed_code runs without errors
        # but its stdout/stderr streams are effectively consumed by our _read_stream_limited mock.
        mock_proc = MagicMock(returncode=0, wait=AsyncMock(return_value=0), kill=MagicMock())
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.drain = AsyncMock()
        mock_proc.stdin.close = MagicMock()

        with patch('asyncio.create_subprocess_exec', return_value=mock_proc):
            response = await code_node_applet.on_message(message)

            # Removed debug prints after diagnosis

            assert response.metadata["status"] == "success"
            assert response.content["ok"] is True
            assert response.content["stdout_truncated"] is True
            assert response.content["stdout"] == "" # Cleaned stdout should be empty
            assert response.content["result"] == 1 # Result should still be extracted
            # _read_stream_limited should have been called for stdout and stderr
            assert mock_read_stream_limited.call_count == 2
            # Check calls for stdout, specifically that max_output_bytes was passed
            assert mock_read_stream_limited.call_args_list[0].args[1] == max_output_configured


@pytest.mark.asyncio
async def test_code_node_runtime_not_found(code_node_applet):
    """Test handling when the runtime executable is not found."""
    # Patch asyncio.create_subprocess_exec directly to simulate FileNotFoundError
    with patch('asyncio.create_subprocess_exec', side_effect=FileNotFoundError):
        message = AppletMessage(
            content={},
            # Use a valid language to pass CodeNodeConfigModel validation,
            # but then simulate runtime not found during subprocess creation
            metadata={"node_data": {"language": "python", "code": "print('hello')"}}
        )
        response = await code_node_applet.on_message(message)

        assert response.metadata["status"] == "error"
        assert response.content["ok"] is False # This should now exist in content
        assert "Runtime not found" in response.content["error"]["message"]
