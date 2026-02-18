"""
Tests for the SynApps Orchestrator models.
"""
import pytest
import time
from datetime import datetime, timezone
from apps.orchestrator.models import (
    CodeNodeConfigModel, LLMNodeConfigModel, LLMMessageModel, LLMRequestModel,
    LLMResponseModel, LLMUsageModel, LLMModelInfoModel, LLMProviderInfoModel,
    ImageGenNodeConfigModel, ImageGenRequestModel, ImageGenResponseModel,
    ImageModelInfoModel, ImageProviderInfoModel, MemoryNodeConfigModel,
    MemorySearchResultModel, HTTPRequestNodeConfigModel, TransformNodeConfigModel,
    IfElseNodeConfigModel, MergeNodeConfigModel, ForEachNodeConfigModel,
    WorkflowRunStatusModel, AuthRegisterRequestModel, AuthLoginRequestModel,
    AuthRefreshRequestModel, APIKeyCreateRequestModel,
    Flow, FlowNode, FlowEdge, User, RefreshToken, UserAPIKey, WorkflowRun
)

# Helper for ORM models (since we don't have a live DB session in unit tests)
class MockORM:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.nodes = []
        self.edges = []
        self.refresh_tokens = []
        self.api_keys = []

    def to_dict(self):
        # Fallback for models without a specific to_dict for testing
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

@pytest.fixture(autouse=True)
def mock_time_time(monkeypatch):
    """Fixture to mock time.time() for consistent timestamps."""
    fixed_time = 1678886400.0  # Mar 15, 2023 12:00:00 UTC
    monkeypatch.setattr(time, 'time', lambda: fixed_time)

def test_workflow_run_status_model_dump_results_none():
    """Test WorkflowRunStatusModel.model_dump handles None results."""
    status_model = WorkflowRunStatusModel(
        run_id="test_run_id",
        flow_id="test_flow_id",
        status="running",
        start_time=time.time(),
        results=None  # Explicitly set to None
    )
    dumped = status_model.model_dump()
    assert "results" in dumped
    assert dumped["results"] == {}
    assert dumped["run_id"] == "test_run_id"

def test_workflow_run_status_model_dump_results_empty():
    """Test WorkflowRunStatusModel.model_dump handles empty results dict."""
    status_model = WorkflowRunStatusModel(
        run_id="test_run_id",
        flow_id="test_flow_id",
        status="running",
        start_time=time.time(),
        results={}
    )
    dumped = status_model.model_dump()
    assert "results" in dumped
    assert dumped["results"] == {}

def test_workflow_run_status_model_dump_results_with_data():
    """Test WorkflowRunStatusModel.model_dump preserves results data."""
    status_model = WorkflowRunStatusModel(
        run_id="test_run_id",
        flow_id="test_flow_id",
        status="running",
        start_time=time.time(),
        results={"node1": {"output": "hello"}}
    )
    dumped = status_model.model_dump()
    assert "results" in dumped
    assert dumped["results"] == {"node1": {"output": "hello"}}

# --- ORM Model to_dict methods ---

def test_user_to_dict():
    user = MockORM(
        id="user123",
        email="test@example.com",
        password_hash="hashed_password",
        is_active=True,
        created_at=time.time(),
        updated_at=time.time(),
    )
    # Patch the to_dict method to match the actual ORM method from models.py
    user.to_dict = User.to_dict.__get__(user, User)
    
    user_dict = user.to_dict()
    assert user_dict == {
        "id": "user123",
        "email": "test@example.com",
        "is_active": True,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    assert "password_hash" not in user_dict

def test_user_api_key_to_dict():
    api_key = MockORM(
        id="key456",
        user_id="user123",
        name="my_key",
        key_prefix="prefix123",
        encrypted_key="encrypted_value",
        is_active=True,
        created_at=time.time(),
        last_used_at=None,
    )
    # Patch the to_dict method
    api_key.to_dict = UserAPIKey.to_dict.__get__(api_key, UserAPIKey)

    api_key_dict = api_key.to_dict()
    assert api_key_dict == {
        "id": "key456",
        "user_id": "user123",
        "name": "my_key",
        "key_prefix": "prefix123",
        "is_active": True,
        "created_at": time.time(),
        "last_used_at": None,
    }
    assert "encrypted_key" not in api_key_dict

def test_flow_to_dict():
    node1 = MockORM(
        id="node1", type="start", position_x=0, position_y=0, data={"label": "Start"}
    )
    node1.to_dict = FlowNode.to_dict.__get__(node1, FlowNode)

    edge1 = MockORM(id="edge1", source="node1", target="node2", animated=False)
    edge1.to_dict = FlowEdge.to_dict.__get__(edge1, FlowEdge)

    flow = MockORM(
        id="flow789",
        name="My Test Flow",
        nodes=[node1],
        edges=[edge1],
    )
    flow.to_dict = Flow.to_dict.__get__(flow, Flow)

    flow_dict = flow.to_dict()
    assert flow_dict["id"] == "flow789"
    assert flow_dict["name"] == "My Test Flow"
    assert len(flow_dict["nodes"]) == 1
    assert flow_dict["nodes"][0]["id"] == "node1"
    assert len(flow_dict["edges"]) == 1
    assert flow_dict["edges"][0]["id"] == "edge1"

def test_flow_node_to_dict():
    node = MockORM(
        id="node1", type="start", position_x=10.0, position_y=20.0, data={"label": "Start"}
    )
    node.to_dict = FlowNode.to_dict.__get__(node, FlowNode)

    node_dict = node.to_dict()
    assert node_dict == {
        "id": "node1",
        "type": "start",
        "position": {"x": 10.0, "y": 20.0},
        "data": {"label": "Start"},
    }

def test_flow_edge_to_dict():
    edge = MockORM(id="edge1", source="node1", target="node2", animated=True)
    edge.to_dict = FlowEdge.to_dict.__get__(edge, FlowEdge)

    edge_dict = edge.to_dict()
    assert edge_dict == {
        "id": "edge1",
        "source": "node1",
        "target": "node2",
        "animated": True,
    }

def test_workflow_run_to_dict():
    run = MockORM(
        id="run123",
        flow_id="flow789",
        status="success",
        current_applet="code",
        progress=100,
        total_steps=100,
        start_time=time.time(),
        end_time=time.time() + 10,
        results={"node1": {"output": "done"}},
        error=None,
        error_details=None,
        input_data={"input": "test"},
        completed_applets=["node1", "node2"],
    )
    run.to_dict = WorkflowRun.to_dict.__get__(run, WorkflowRun)

    run_dict = run.to_dict()
    assert run_dict["run_id"] == "run123"
    assert run_dict["flow_id"] == "flow789"
    assert run_dict["status"] == "success"
    assert run_dict["results"] == {"node1": {"output": "done"}}
    assert run_dict["completed_applets"] == ["node1", "node2"]


# --- Pydantic Model field_validators and default behaviors ---

def test_auth_register_request_model_email_validation():
    # Valid email
    model = AuthRegisterRequestModel(email="test@example.com", password="password123")
    assert model.email == "test@example.com"

    # Email with leading/trailing spaces and mixed case
    model = AuthRegisterRequestModel(email="  Test@EXAMPLE.COM  ", password="password123")
    assert model.email == "test@example.com"

    # Invalid email - missing @
    with pytest.raises(ValueError, match="email must be a valid email address"):
        AuthRegisterRequestModel(email="invalid-email", password="password123")

    # Invalid email - @ at start
    with pytest.raises(ValueError, match="email must be a valid email address"):
        AuthRegisterRequestModel(email="@example.com", password="password123")

    # Invalid email - @ at end
    with pytest.raises(ValueError, match="email must be a valid email address"):
        AuthRegisterRequestModel(email="test@", password="password123")

def test_auth_login_request_model_email_normalization():
    model = AuthLoginRequestModel(email="  Test@EXAMPLE.COM  ", password="password123")
    assert model.email == "test@example.com"

def test_api_key_create_request_model_name_normalization():
    model = APIKeyCreateRequestModel(name="  My API Key  ")
    assert model.name == "My API Key"

    model = APIKeyCreateRequestModel() # Test default value
    assert model.name == "default"

def test_llm_message_model_role_validation():
    LLMMessageModel(role="system", content="hi")
    LLMMessageModel(role="user", content="hi")
    LLMMessageModel(role="assistant", content="hi")
    LLMMessageModel(role="tool", content="hi")

    with pytest.raises(ValueError, match="role must be one of: system, user, assistant, tool"):
        LLMMessageModel(role="invalid", content="hi")

def test_llm_node_config_model_provider_validation():
    LLMNodeConfigModel(provider="openai")
    LLMNodeConfigModel(provider="anthropic")
    LLMNodeConfigModel(provider="google")
    LLMNodeConfigModel(provider="ollama")
    LLMNodeConfigModel(provider="custom")

    with pytest.raises(ValueError, match="provider must be one of"):
        LLMNodeConfigModel(provider="invalid")

def test_code_node_config_model_language_validation():
    CodeNodeConfigModel(language="python")
    CodeNodeConfigModel(language="javascript")
    CodeNodeConfigModel(language="py") # alias
    CodeNodeConfigModel(language="python3") # alias
    CodeNodeConfigModel(language="js") # alias
    CodeNodeConfigModel(language="node") # alias
    CodeNodeConfigModel(language="nodejs") # alias

    with pytest.raises(ValueError, match="language must be one of"):
        CodeNodeConfigModel(language="ruby")

def test_code_node_config_model_working_dir_validation():
    model = CodeNodeConfigModel(working_dir="/tmp/subdir")
    assert model.working_dir == "/tmp/subdir"

    model = CodeNodeConfigModel(working_dir="/tmp")
    assert model.working_dir == "/tmp"

    model = CodeNodeConfigModel(working_dir="  /tmp/another  ")
    assert model.working_dir == "/tmp/another"

    model = CodeNodeConfigModel(working_dir="  ") # empty after strip, defaults to /tmp
    assert model.working_dir == "/tmp"

    with pytest.raises(ValueError, match="working_dir must be under /tmp"):
        CodeNodeConfigModel(working_dir="/var/log")

    with pytest.raises(ValueError, match="working_dir must be under /tmp"):
        CodeNodeConfigModel(working_dir="../tmp") # even if it resolves to /tmp, current check is on normalized

def test_image_gen_node_config_model_provider_validation():
    ImageGenNodeConfigModel(provider="openai")
    ImageGenNodeConfigModel(provider="stability")
    ImageGenNodeConfigModel(provider="flux")

    with pytest.raises(ValueError, match="provider must be one of"):
        ImageGenNodeConfigModel(provider="invalid")

def test_image_gen_node_config_model_response_format_validation():
    ImageGenNodeConfigModel(response_format="b64_json")
    ImageGenNodeConfigModel(response_format="url")

    with pytest.raises(ValueError, match="response_format must be one of: b64_json, url"):
        ImageGenNodeConfigModel(response_format="png")

def test_image_gen_request_model_response_format_validation():
    ImageGenRequestModel(prompt="test", model="model1", response_format="b64_json")
    ImageGenRequestModel(prompt="test", model="model1", response_format="url")

    with pytest.raises(ValueError, match="response_format must be one of: b64_json, url"):
        ImageGenRequestModel(prompt="test", model="model1", response_format="jpeg")

def test_memory_node_config_model_operation_validation():
    MemoryNodeConfigModel(operation="store")
    MemoryNodeConfigModel(operation="retrieve")
    MemoryNodeConfigModel(operation="delete")
    MemoryNodeConfigModel(operation="clear")

    with pytest.raises(ValueError, match="operation must be one of"):
        MemoryNodeConfigModel(operation="search")

def test_memory_node_config_model_backend_validation():
    MemoryNodeConfigModel(backend="sqlite_fts")
    MemoryNodeConfigModel(backend="chroma")

    with pytest.raises(ValueError, match="backend must be one of"):
        MemoryNodeConfigModel(backend="pinecone")

def test_memory_node_config_model_query_normalization():
    model = MemoryNodeConfigModel(query="  some query  ")
    assert model.query == "some query"

    model = MemoryNodeConfigModel(query="  ")
    assert model.query is None

    model = MemoryNodeConfigModel(query=None)
    assert model.query is None

def test_memory_node_config_model_tags_normalization():
    model = MemoryNodeConfigModel(tags=["  tag1  ", "tag2", "TAG1", ""])
    assert model.tags == ["tag1", "tag2"] # Unique and stripped

def test_http_request_node_config_model_url_validation():
    model = HTTPRequestNodeConfigModel(url="http://example.com")
    assert model.url == "http://example.com"

    with pytest.raises(ValueError, match="url cannot be blank"):
        HTTPRequestNodeConfigModel(url="  ")

    with pytest.raises(ValueError, match="url cannot be blank"):
        HTTPRequestNodeConfigModel(url="")

def test_http_request_node_config_model_method_validation():
    HTTPRequestNodeConfigModel(url="http://example.com", method="GET")
    HTTPRequestNodeConfigModel(url="http://example.com", method="POST")
    HTTPRequestNodeConfigModel(url="http://example.com", method="put") # Lower case
    HTTPRequestNodeConfigModel(url="http://example.com", method="DELETE")

    with pytest.raises(ValueError, match="method must be one of"):
        HTTPRequestNodeConfigModel(url="http://example.com", method="PATCH")

def test_http_request_node_config_model_body_type_validation():
    HTTPRequestNodeConfigModel(url="http://example.com", body_type="auto")
    HTTPRequestNodeConfigModel(url="http://example.com", body_type="json")
    HTTPRequestNodeConfigModel(url="http://example.com", body_type="text")
    HTTPRequestNodeConfigModel(url="http://example.com", body_type="form")
    HTTPRequestNodeConfigModel(url="http://example.com", body_type="none")
    HTTPRequestNodeConfigModel(url="http://example.com", body_type=" AuTo ")

    with pytest.raises(ValueError, match="body_type must be one of"):
        HTTPRequestNodeConfigModel(url="http://example.com", body_type="xml")

def test_transform_node_config_model_operation_validation():
    TransformNodeConfigModel(operation="json_path")
    TransformNodeConfigModel(operation="template")
    TransformNodeConfigModel(operation="regex_replace")
    TransformNodeConfigModel(operation="split_join")
    TransformNodeConfigModel(operation=" jsonpath ") # alias and strip
    TransformNodeConfigModel(operation="template-string") # alias and hyphen

    with pytest.raises(ValueError, match="operation must be one of"):
        TransformNodeConfigModel(operation="parse")

def test_transform_node_config_model_json_path_normalization():
    model = TransformNodeConfigModel(json_path="foo.bar")
    assert model.json_path == "$.foo.bar"

    model = TransformNodeConfigModel(json_path="$[0]")
    assert model.json_path == "$[0]"

    model = TransformNodeConfigModel(json_path="  ")
    assert model.json_path == "$"

def test_transform_node_config_model_regex_flags_validation():
    TransformNodeConfigModel(regex_flags="im")
    TransformNodeConfigModel(regex_flags="iMsX") # Mixed case, will normalize and unique
    model = TransformNodeConfigModel(regex_flags="iix")
    assert model.regex_flags == "ix"

    with pytest.raises(ValueError, match="regex_flags may contain only: i, m, s, x"):
        TransformNodeConfigModel(regex_flags="z")

def test_if_else_node_config_model_operation_validation():
    IfElseNodeConfigModel(operation="contains")
    IfElseNodeConfigModel(operation="equals")
    IfElseNodeConfigModel(operation="regex")
    IfElseNodeConfigModel(operation="json_path")
    IfElseNodeConfigModel(operation="eq") # alias

    with pytest.raises(ValueError, match="operation must be one of"):
        IfElseNodeConfigModel(operation="greater_than")

def test_if_else_node_config_model_regex_flags_validation():
    IfElseNodeConfigModel(regex_flags="i")
    model = IfElseNodeConfigModel(regex_flags="iMs")
    assert model.regex_flags == "ims"

    with pytest.raises(ValueError, match="regex_flags may contain only: i, m, s, x"):
        IfElseNodeConfigModel(regex_flags="a")

def test_if_else_node_config_model_json_path_normalization():
    model = IfElseNodeConfigModel(json_path="foo")
    assert model.json_path == "$.foo"

    model = IfElseNodeConfigModel(json_path="  ")
    assert model.json_path == "$"

def test_if_else_node_config_model_target_normalization():
    model = IfElseNodeConfigModel(true_target="  target_id  ", false_target="")
    assert model.true_target == "target_id"
    assert model.false_target is None

    model = IfElseNodeConfigModel(true_target=None)
    assert model.true_target is None

def test_merge_node_config_model_strategy_validation():
    MergeNodeConfigModel(strategy="concatenate")
    MergeNodeConfigModel(strategy="array")
    MergeNodeConfigModel(strategy="first_wins")
    MergeNodeConfigModel(strategy="concat") # alias

    with pytest.raises(ValueError, match="strategy must be one of"):
        MergeNodeConfigModel(strategy="sum")

def test_for_each_node_config_model_defaults():
    model = ForEachNodeConfigModel()
    assert model.label == "For-Each"
    assert model.array_source == "{{input}}"
    assert model.max_iterations == 1000
    assert model.parallel is False
    assert model.concurrency_limit == 10

def test_flow_node_model_defaults():
    model = FlowNodeModel(id="node1", type="test", position={"x": 10, "y": 20})
    assert model.data == {}

def test_flow_edge_model_defaults():
    model = FlowEdgeModel(id="edge1", source="node1", target="node2")
    assert model.animated is False

def test_flow_model_defaults():
    model = FlowModel(name="test flow")
    assert model.nodes == []
    assert model.edges == []

def test_llm_request_model_defaults():
    model = LLMRequestModel(model="gpt-4")
    assert model.messages == []
    assert model.temperature == 0.7
    assert model.max_tokens == 1024
    assert model.top_p == 1.0
    assert model.stop_sequences == []
    assert model.stream is False
    assert model.structured_output is False
    assert model.extra == {}

def test_llm_response_model_defaults():
    model = LLMResponseModel(content="test", model="gpt-4", provider="openai")
    assert model.usage.prompt_tokens == 0
    assert model.usage.completion_tokens == 0
    assert model.usage.total_tokens == 0
    assert model.finish_reason == "stop"
    assert model.raw == {}

def test_llm_model_info_model_defaults():
    model = LLMModelInfoModel(id="model1", name="Model 1", provider="openai")
    assert model.context_window == 0
    assert model.supports_streaming is True
    assert model.supports_vision is False
    assert model.max_output_tokens is None

def test_llm_provider_info_model_defaults():
    model = LLMProviderInfoModel(name="openai", configured=True)
    assert model.reason == ""
    assert model.models == []

def test_image_gen_request_model_defaults():
    model = ImageGenRequestModel(prompt="test", model="dall-e")
    assert model.negative_prompt == ""
    assert model.size == "1024x1024"
    assert model.style == "photorealistic"
    assert model.quality == "standard"
    assert model.n == 1
    assert model.response_format == "b64_json"
    assert model.extra == {}

def test_image_gen_response_model_defaults():
    model = ImageGenResponseModel(model="dall-e", provider="openai")
    assert model.images == []
    assert model.revised_prompt is None
    assert model.raw == {}

def test_image_model_info_model_defaults():
    model = ImageModelInfoModel(id="dall-e", name="Dall-E", provider="openai")
    assert model.supports_base64 is True
    assert model.supports_url is True
    assert model.max_images == 1

def test_image_provider_info_model_defaults():
    model = ImageProviderInfoModel(name="openai", configured=True)
    assert model.reason == ""
    assert model.models == []

def test_memory_search_result_model_defaults():
    model = MemorySearchResultModel(key="key1", data="value1")
    assert model.score == 0.0
    assert model.metadata == {}

def test_http_request_node_config_model_defaults():
    model = HTTPRequestNodeConfigModel(url="http://test.com")
    assert model.method == "GET"
    assert model.headers == {}
    assert model.query_params == {}
    assert model.body_template is None
    assert model.body_type == "auto"
    assert model.timeout_seconds == 30.0
    assert model.allow_redirects is True
    assert model.verify_ssl is True
    assert model.include_response_headers is True
    assert model.extra == {}

def test_transform_node_config_model_defaults():
    model = TransformNodeConfigModel()
    assert model.label == "Transform"
    assert model.operation == "template"
    assert model.source == "{{content}}"
    assert model.json_path == "$"
    assert model.template == "{{source}}"
    assert model.regex_pattern == ""
    assert model.regex_replacement == ""
    assert model.regex_flags == ""
    assert model.regex_count == 0
    assert model.split_delimiter == ","
    assert model.split_maxsplit == -1
    assert model.split_index is None
    assert model.return_list is False
    assert model.strip_items is False
    assert model.drop_empty is False
    assert model.extra == {}

def test_if_else_node_config_model_defaults():
    model = IfElseNodeConfigModel()
    assert model.label == "If / Else"
    assert model.operation == "equals"
    assert model.source == "{{content}}"
    assert model.value is None
    assert model.case_sensitive is False
    assert model.negate is False
    assert model.regex_pattern == ""
    assert model.regex_flags == ""
    assert model.json_path == "$"
    assert model.true_target is None
    assert model.false_target is None
    assert model.extra == {}

def test_merge_node_config_model_defaults():
    model = MergeNodeConfigModel()
    assert model.label == "Merge"
    assert model.strategy == "array"
    assert model.delimiter == '\n' # Corrected this line
    assert model.extra == {}
