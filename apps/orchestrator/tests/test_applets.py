import pytest
import os
import json
from unittest.mock import patch, MagicMock, AsyncMock
from apps.applets.writer.applet import WriterApplet
from apps.applets.artist.applet import ArtistApplet
from apps.applets.memory.applet import MemoryApplet
from apps.orchestrator.main import AppletMessage

@pytest.mark.asyncio
async def test_writer_applet_success():
    # Mock environment variable
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
        applet = WriterApplet()
        
        # Test string input
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Generated text"}}]
        }
        
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            
            message = AppletMessage(content="Test prompt", context={}, metadata={})
            response = await applet.on_message(message)
            assert response.content == "Generated text"
            
            # Test dict input with 'prompt'
            message = AppletMessage(content={"prompt": "Test prompt 2"}, context={}, metadata={})
            response = await applet.on_message(message)
            assert response.content == "Generated text"

            # Test dict input with 'text'
            message = AppletMessage(content={"text": "Test text"}, context={}, metadata={})
            response = await applet.on_message(message)
            assert response.content == "Generated text"

            # Test missing input (should use default)
            message = AppletMessage(content={}, context={}, metadata={})
            response = await applet.on_message(message)
            assert response.content == "Generated text"

@pytest.mark.asyncio
async def test_writer_applet_no_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(RuntimeError):
            WriterApplet()

@pytest.mark.asyncio
async def test_writer_applet_error():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
        applet = WriterApplet()
        
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            
            message = AppletMessage(content="Test prompt", context={}, metadata={})
            response = await applet.on_message(message)
            assert "Error generating text" in response.content

@pytest.mark.asyncio
async def test_artist_applet_stability_success():
    with patch.dict(os.environ, {"STABILITY_API_KEY": "test_key"}):
        applet = ArtistApplet()
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "artifacts": [{"base64": "fake_image_data"}]
        }
        
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            
            message = AppletMessage(content="Test prompt", context={"image_generator": "stability"}, metadata={})
            response = await applet.on_message(message)
            assert response.content["image"] == "fake_image_data"
            assert response.content["generator"] == "stability"

@pytest.mark.asyncio
async def test_artist_applet_openai_success():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
        applet = ArtistApplet()
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"b64_json": "fake_openai_image"}]
        }
        
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            
            message = AppletMessage(content="Test prompt", context={"image_generator": "openai"}, metadata={})
            response = await applet.on_message(message)
            assert response.content["image"] == "fake_openai_image"
            assert response.content["generator"] == "openai"

@pytest.mark.asyncio
async def test_artist_applet_no_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(RuntimeError):
            ArtistApplet()

@pytest.mark.asyncio
async def test_memory_applet_store_retrieve():
    applet = MemoryApplet()
    
    # Test Store
    store_message = AppletMessage(
        content={"operation": "store", "data": {"info": "important"}, "key": "my_key"},
        context={},
        metadata={}
    )
    store_response = await applet.on_message(store_message)
    assert store_response.content["status"] == "stored"
    assert store_response.content["key"] == "my_key"
    
    # Test Retrieve
    retrieve_message = AppletMessage(
        content={"operation": "retrieve", "key": "my_key"},
        context={},
        metadata={}
    )
    retrieve_response = await applet.on_message(retrieve_message)
    assert retrieve_response.content["info"] == "important"

@pytest.mark.asyncio
async def test_memory_applet_retrieve_by_tags():
    applet = MemoryApplet()
    
    # Store with tags
    await applet.on_message(AppletMessage(
        content={"operation": "store", "data": "val1", "key": "k1", "tags": ["tag1"]},
        context={}, metadata={}
    ))
    await applet.on_message(AppletMessage(
        content={"operation": "store", "data": "val2", "key": "k2", "tags": ["tag2"]},
        context={}, metadata={}
    ))
    
    # Retrieve by tag1
    retrieve_message = AppletMessage(
        content={"operation": "retrieve", "tags": ["tag1"]},
        context={},
        metadata={}
    )
    response = await applet.on_message(retrieve_message)
    assert "k1" in response.content["memories"]
    assert "k2" not in response.content["memories"]

@pytest.mark.asyncio
async def test_artist_applet_input_variations():
    with patch.dict(os.environ, {"STABILITY_API_KEY": "test_key"}):
        applet = ArtistApplet()
        with patch("apps.applets.artist.applet.ArtistApplet._call_stability_api", new_callable=AsyncMock) as mock_stability:
            mock_stability.return_value = "img"
            
            # Test dict with 'text'
            await applet.on_message(AppletMessage(content={"text": "some text"}, context={}, metadata={}))
            assert mock_stability.call_args[0][0] == "some text"

@pytest.mark.asyncio
async def test_memory_applet_invalid_operation():
    applet = MemoryApplet()
    message = AppletMessage(content={"operation": "invalid"}, context={}, metadata={})
    response = await applet.on_message(message)
    assert response.metadata["status"] == "error"
    assert "Invalid operation" in response.content["error"]

@pytest.mark.asyncio
async def test_memory_applet_clear():
    applet = MemoryApplet()
    await applet.on_message(AppletMessage(content={"operation": "store", "key": "k1", "data": "v1"}, context={}, metadata={}))
    await applet.on_message(AppletMessage(content={"operation": "clear"}, context={}, metadata={}))
    
    response = await applet.on_message(AppletMessage(content={"operation": "retrieve", "key": "k1"}, context={}, metadata={}))
    assert response.content["status"] == "not_found"

@pytest.mark.asyncio
async def test_memory_applet_delete():
    applet = MemoryApplet()
    await applet.on_message(AppletMessage(content={"operation": "store", "key": "k1", "data": "v1"}, context={}, metadata={}))
    
    # Delete existing
    response = await applet.on_message(AppletMessage(content={"operation": "delete", "key": "k1"}, context={}, metadata={}))
    assert response.content["status"] == "deleted"
    
    # Delete non-existing
    response = await applet.on_message(AppletMessage(content={"operation": "delete", "key": "k1"}, context={}, metadata={}))
    assert response.content["status"] == "not_found"

@pytest.mark.asyncio
async def test_artist_applet_variations():
    with patch.dict(os.environ, {"STABILITY_API_KEY": "test_key", "OPENAI_API_KEY": "test_key"}):
        applet = ArtistApplet()
        
        # Test fallback Stability -> OpenAI
        with patch("apps.applets.artist.applet.ArtistApplet._call_stability_api", new_callable=AsyncMock) as mock_stability:
            mock_stability.side_effect = Exception("Stability failed")
            with patch("apps.applets.artist.applet.ArtistApplet._call_openai_api", new_callable=AsyncMock) as mock_openai:
                mock_openai.return_value = "openai_img"
                
                response = await applet.on_message(AppletMessage(content="test", context={"image_generator": "stability"}, metadata={}))
                assert response.content["image"] == "openai_img"
                assert response.content["generator"] == "openai"

        # Test OpenAI failure
        with patch("apps.applets.artist.applet.ArtistApplet._call_openai_api", new_callable=AsyncMock) as mock_openai:
            mock_openai.side_effect = Exception("OpenAI failed")
            response = await applet.on_message(AppletMessage(content="test", context={"image_generator": "openai"}, metadata={}))
            assert response.content["generator"] == "error"

@pytest.mark.asyncio
async def test_artist_applet_mock():
    # Test with no API keys (mock)
    with patch.dict(os.environ, {"STABILITY_API_KEY": "test"}):
        applet = ArtistApplet()
        # Clear keys after init to test the fallback to mock in _generate_image
        applet.stability_api_key = None
        applet.openai_api_key = None
        
        response = await applet.on_message(AppletMessage(content="test", context={}, metadata={}))
        assert response.content["generator"] == "mock"

@pytest.mark.asyncio
async def test_writer_applet_with_system_prompt():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
        applet = WriterApplet()
        
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"choices": [{"message": {"content": "response"}}]}
            mock_post.return_value = mock_response
            
            message = AppletMessage(content="test", context={"system_prompt": "custom prompt"}, metadata={})
            await applet.on_message(message)
            
            # Verify system prompt was passed to API
            args, kwargs = mock_post.call_args
            assert kwargs["json"]["messages"][0]["content"] == "custom prompt"

@pytest.mark.asyncio
async def test_writer_applet_mock_fallback():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
        applet = WriterApplet()
        applet.api_key = None # Clear after init
        
        message = AppletMessage(content="test", context={}, metadata={})
        response = await applet.on_message(message)
        assert "mock response" in response.content

@pytest.mark.asyncio
async def test_artist_applet_style_and_errors():
    with patch.dict(os.environ, {"STABILITY_API_KEY": "test_key", "OPENAI_API_KEY": "test_key"}):
        applet = ArtistApplet()
        
        # Test style
        with patch("apps.applets.artist.applet.ArtistApplet._call_stability_api", new_callable=AsyncMock) as mock_stability:
            mock_stability.return_value = "img"
            await applet.on_message(AppletMessage(content="test", context={"style": "cyberpunk"}, metadata={}))
            assert mock_stability.call_args[0][1] == "cyberpunk"

        # Test stability API non-200
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Error"
            mock_post.return_value = mock_response
            
            with pytest.raises(Exception):
                await applet._call_stability_api("prompt", "style")

        # Test openai API non-200
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Error"
            mock_post.return_value = mock_response
            
            with pytest.raises(Exception):
                await applet._call_openai_api("prompt", "style")

@pytest.mark.asyncio
async def test_memory_applet_not_found():
    applet = MemoryApplet()
    retrieve_message = AppletMessage(
        content={"operation": "retrieve", "key": "non_existent"},
        context={},
        metadata={}
    )
    response = await applet.on_message(retrieve_message)
    assert response.content["status"] == "not_found"
