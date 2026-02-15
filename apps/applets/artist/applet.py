"""
ArtistApplet - Image generation applet for SynApps

This applet uses Stable Diffusion or dall-e-3 to generate images from text prompts.
"""
import os
import base64
import json
import httpx
import logging
from typing import Dict, Any, Optional

# Import base applet from orchestrator
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'orchestrator'))
from main import BaseApplet, AppletMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("artist-applet")

class ArtistApplet(BaseApplet):
    """
    Artist Applet that generates images from text prompts.
    
    Capabilities:
    - Image generation from text descriptions
    - Style transfer
    - Visual creation
    """
    
    VERSION = "0.1.0"
    CAPABILITIES = ["image-generation", "text-to-image", "visual-creation"]
    
    def __init__(self):
        """Initialize the Artist Applet."""
        self.stability_api_key = os.environ.get("STABILITY_API_KEY")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        
        if not (self.stability_api_key or self.openai_api_key):
            logger.error("Neither STABILITY_API_KEY nor OPENAI_API_KEY environment variables are set. ArtistApplet cannot function without at least one.")
            raise RuntimeError("At least one of STABILITY_API_KEY or OPENAI_API_KEY environment variables is required for ArtistApplet. Please set before starting the app.")
    
    async def on_message(self, message: AppletMessage) -> AppletMessage:
        """Process an incoming message and generate an image."""
        logger.info("Artist Applet received message")
        
        # Extract content from message
        content = message.content
        context = message.context
        
        # Get the input text/prompt
        prompt = ""
        if isinstance(content, str):
            prompt = content
        elif isinstance(content, dict) and "prompt" in content:
            prompt = content["prompt"]
        elif isinstance(content, dict) and "text" in content:
            prompt = content["text"]
        else:
            prompt = "A beautiful landscape with mountains and a lake."
        
        # Get generator options
        generator = context.get("image_generator", "stability")  # Default to Stability AI
        
        # Get style options
        style = context.get("style", "photorealistic")
        
        # Generate the image
        image_data, generator_used = await self._generate_image(prompt, generator, style)
        
        # Return the generated image
        return AppletMessage(
            content={
                "image": image_data,
                "prompt": prompt,
                "generator": generator_used
            },
            context={**context},  # Preserve context
            metadata={"applet": "artist", "generator": generator_used}
        )
    
    async def _generate_image(self, prompt: str, generator: str, style: str) -> tuple[str, str]:
        """Generate an image using the specified API."""
        # For MVP, we'll mock the response
        if not (self.stability_api_key or self.openai_api_key):
            # Return mock base64 image data (1x1 transparent pixel)
            return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=", "mock"
        
        # Try Stability AI first if specified
        if generator.lower() == "stability" and self.stability_api_key:
            try:
                return await self._call_stability_api(prompt, style), "stability"
            except Exception as e:
                logger.error(f"Error with Stability API: {e}, falling back to OpenAI")
                generator = "openai"  # Fall back to OpenAI
        
        # Use dall-e-3 if specified or as fallback
        if generator.lower() == "openai" and self.openai_api_key:
            try:
                return await self._call_openai_api(prompt, style), "openai"
            except Exception as e:
                logger.error(f"Error with OpenAI API: {e}")
                # Return error image
                return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=", "error"
        
        # If no valid generator or API key, return mock
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=", "mock"
    
    async def _call_stability_api(self, prompt: str, style: str) -> str:
        """Call Stability AI API to generate an image."""
        try:
            engine_id = "stable-diffusion-xl-1024-v1-0"
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"https://api.stability.ai/v1/generation/{engine_id}/text-to-image",
                    headers={
                        "Authorization": f"Bearer {self.stability_api_key}",
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    },
                    json={
                        "text_prompts": [
                            {
                                "text": f"{prompt}, {style} style",
                                "weight": 1.0
                            }
                        ],
                        "cfg_scale": 7,
                        "height": 1024,
                        "width": 1024,
                        "samples": 1,
                        "steps": 30
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"Stability API error: {response.text}")
                    raise Exception(f"Stability API error: {response.text}")
                
                data = response.json()
                
                # Extract the base64 image
                if "artifacts" in data and len(data["artifacts"]) > 0:
                    return data["artifacts"][0]["base64"]
                else:
                    raise Exception("No image generated by Stability API")
                
        except Exception as e:
            logger.error(f"Error calling Stability API: {e}")
            raise
    
    async def _call_openai_api(self, prompt: str, style: str) -> str:
        """Call OpenAI dall-e-3 API to generate an image."""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/images/generations",
                    headers={
                        "Authorization": f"Bearer {self.openai_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "dall-e-3-3",
                        "prompt": f"{prompt}, {style} style",
                        "n": 1,
                        "size": "1024x1024",
                        "response_format": "b64_json"
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"OpenAI API error: {response.text}")
                    raise Exception(f"OpenAI API error: {response.text}")
                
                data = response.json()
                
                # Extract the base64 image
                if "data" in data and len(data["data"]) > 0 and "b64_json" in data["data"][0]:
                    return data["data"][0]["b64_json"]
                else:
                    raise Exception("No image generated by OpenAI API")
                
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise

# For testing
if __name__ == "__main__":
    import asyncio
    
    async def test_artist():
        applet = ArtistApplet()
        message = AppletMessage(
            content="A futuristic city with flying cars and neon lights",
            context={},
            metadata={}
        )
        response = await applet.on_message(message)
        print(f"Generated image with {response.content['generator']}")
    
    asyncio.run(test_artist())
