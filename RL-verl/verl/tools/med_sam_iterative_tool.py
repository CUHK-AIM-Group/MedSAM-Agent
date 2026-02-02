# Copyright 2025
# Iterative segmentation tools for MedSAM-Agent recipe
# Three separate tools: add_positive_point, add_negative_point, stop_action

import logging
import os
from typing import Any, Optional
from uuid import uuid4
import base64
from io import BytesIO

import requests
import numpy as np
from PIL import Image

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse
from qwen_vl_utils import fetch_image

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# Shared state manager for multi-turn segmentation sessions
# This allows the three tools to share the same API session
class SegmentationSessionManager:
    """Singleton manager for sharing segmentation sessions across tools."""
    _instance = None
    _sessions = {}  # instance_id -> session_data
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_session(self, instance_id: str) -> Optional[dict]:
        return self._sessions.get(instance_id)
    
    def set_session(self, instance_id: str, session_data: dict):
        self._sessions[instance_id] = session_data
    
    def delete_session(self, instance_id: str):
        if instance_id in self._sessions:
            del self._sessions[instance_id]
    
    def update_clicks(self, instance_id: str, point: list, label: int):
        """Add a new click to the session."""
        session = self._sessions.get(instance_id)
        if session:
            session["clicks_list"].append((point, label))


class BaseSegmentationTool(BaseTool):
    """Base class for segmentation tools with shared API logic."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.session_manager = SegmentationSessionManager()
        
        # API configuration
        self.api_base_url = config.get("api_base_url", "http://localhost:8000")
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1.0)
        self.timeout = config.get("timeout", 30)
        self.use_multiturn = config.get("use_multiturn", True)
        
        # Initialize requests session - keep proxy for external requests
        self.session = requests.Session()
        
        logger.info(f"Initialized {self.__class__.__name__} with API at {self.api_base_url}")
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema
    
    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')
    
    def _decode_base64_to_image(self, base64_str: str) -> Image.Image:
        """Convert base64 string to PIL Image."""
        img_bytes = base64.b64decode(base64_str)
        return Image.open(BytesIO(img_bytes))
    
    async def _create_api_session(self, image: Image.Image) -> str:
        """Create a new API session for multi-turn segmentation."""
        import asyncio
        
        img_buffer = BytesIO()
        image.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        
        files = {'image': ('image.png', img_buffer, 'image/png')}
        
        for attempt in range(self.max_retries):
            try:
                img_buffer.seek(0)
                # Disable proxy only for localhost API calls
                response = self.session.post(
                    f"{self.api_base_url}/session/create",
                    files=files,
                    timeout=self.timeout,
                    proxies={'http': None, 'https': None}
                )
                
                if response.status_code != 200:
                    raise RuntimeError(f"API returned status {response.status_code}: {response.text[:200]}")
                
                result = response.json()
                session_id = result.get("session_id")
                
                if not session_id:
                    raise RuntimeError("API response missing 'session_id' field")
                
                logger.info(f"Created API session: {session_id}")
                return session_id
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} to create session failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise
    
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a new segmentation instance with the input image.
        
        This is shared by all three tools - they all use the same session.
        """
        if instance_id is None:
            instance_id = str(uuid4())
        
        # Check if session already exists (another tool already created it)
        existing_session = self.session_manager.get_session(instance_id)
        if existing_session is not None:
            logger.debug(f"Session {instance_id} already exists, reusing")
            return instance_id, ToolResponse()
        
        # Extract create_kwargs
        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)
        
        image = kwargs.get("image")
        if image is None:
            raise ValueError("Missing required 'image' parameter in kwargs")
        
        # Fetch and process image
        img = fetch_image({"image": image})
        
        # Create API session
        api_session_id = None
        if self.use_multiturn:
            try:
                api_session_id = await self._create_api_session(img)
            except Exception as e:
                logger.warning(f"Failed to create API session: {e}")
        
        # Initialize session data
        session_data = {
            "image": img,
            "clicks_list": [],  # [(point, label), ...]
            "api_session_id": api_session_id,
            "stopped": False,  # Track if stop_action was called
            "last_mask": None,  # Cache last generated mask
        }
        
        self.session_manager.set_session(instance_id, session_data)
        logger.info(f"Created instance {instance_id} with image size {img.size}")
        
        return instance_id, ToolResponse()
    
    async def _predict_with_multiturn_api(
        self, 
        session_data: dict,
    ) -> Image.Image:
        """Perform segmentation using multi-turn session API."""
        import json
        import asyncio
        
        api_session_id = session_data["api_session_id"]
        if not api_session_id:
            raise RuntimeError("No API session ID available")
        
        # Get all accumulated clicks
        all_points = [point for point, _ in session_data["clicks_list"]]
        all_labels = [label for _, label in session_data["clicks_list"]]
        
        if not all_points:
            raise ValueError("No points available for segmentation")
        
        data = {
            'session_id': api_session_id,
            'point_coords': json.dumps(all_points),
            'point_labels': json.dumps(all_labels),
            'use_previous_mask': True
        }
        
        # Retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Disable proxy only for localhost API calls
                response = self.session.post(
                    f"{self.api_base_url}/session/segment",
                    data=data,
                    timeout=self.timeout,
                    proxies={'http': None, 'https': None}
                )
                
                if response.status_code != 200:
                    error_msg = f"API returned status {response.status_code}: {response.text[:200]}"
                    logger.warning(f"Attempt {attempt + 1}/{self.max_retries}: {error_msg}")
                    last_error = error_msg
                    
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay)
                    continue
                
                result = response.json()
                masks_base64 = result.get("masks", [])
                
                if not masks_base64:
                    error_msg = "API response missing masks"
                    logger.warning(error_msg)
                    last_error = error_msg
                    
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay)
                    continue
                
                mask_pil = self._decode_base64_to_image(masks_base64[0])
                logger.debug(f"Got mask from API: {mask_pil.size}, clicks={len(all_points)}")
                
                # Cache the mask
                session_data["last_mask"] = mask_pil
                
                return mask_pil
                
            except Exception as e:
                error_msg = f"Error: {e}"
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries}: {error_msg}")
                last_error = error_msg
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                continue
        
        raise RuntimeError(f"Failed after {self.max_retries} attempts. Last error: {last_error}")
    
    async def release(self, instance_id: str, **kwargs) -> None:
        """Release resources for a specific instance."""
        session_data = self.session_manager.get_session(instance_id)
        if session_data:
            # Clean up API session
            api_session_id = session_data.get("api_session_id")
            if api_session_id:
                try:
                    # Disable proxy only for localhost API calls
                    response = self.session.delete(
                        f"{self.api_base_url}/session/{api_session_id}",
                        timeout=self.timeout,
                        proxies={'http': None, 'https': None}
                    )
                    if response.status_code == 200:
                        logger.debug(f"Deleted API session {api_session_id}")
                except Exception as e:
                    logger.warning(f"Error deleting API session: {e}")
            
            # Remove from manager
            self.session_manager.delete_session(instance_id)
            logger.debug(f"Released instance {instance_id}")


class AddPositivePointTool(BaseSegmentationTool):
    """Tool for adding positive points to expand the mask."""
    
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Add a positive point and generate updated mask.
        
        Expected parameters:
          - x: X coordinate (0-1000)
          - y: Y coordinate (0-1000)
        """
        session_data = self.session_manager.get_session(instance_id)
        if not session_data:
            return ToolResponse(text="Error: invalid instance_id"), -0.1, {"success": False}
        
        # Check if already stopped
        if session_data.get("stopped"):
            return ToolResponse(text="Error: segmentation already stopped"), -0.05, {"success": False}
        
        # Parse coordinates
        x = parameters.get("x")
        y = parameters.get("y")
        
        if x is None or y is None:
            return ToolResponse(text="Error: missing x or y coordinate"), -0.05, {"success": False}
        
        # Convert from 0-1000 to image pixel coordinates
        img = session_data["image"]
        img_width, img_height = img.size
        pixel_x = int(x * img_width / 1000)
        pixel_y = int(y * img_height / 1000)
        
        # Add positive point (label=1)
        point = [pixel_x, pixel_y]
        self.session_manager.update_clicks(instance_id, point, label=1)
        
        # Generate updated mask
        try:
            mask = await self._predict_with_multiturn_api(session_data)
            response_text = f"Added positive point at ({x}, {y}). Mask updated."
            return ToolResponse(image=[mask], text=response_text), 0.0, {"success": True}
        except Exception as e:
            error_msg = f"Failed to generate mask: {e}"
            logger.error(error_msg)
            return ToolResponse(text=f"Error: {error_msg}"), -0.1, {"success": False}


class AddNegativePointTool(BaseSegmentationTool):
    """Tool for adding negative points to refine the mask."""
    
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Add a negative point and generate updated mask.
        
        Expected parameters:
          - x: X coordinate (0-1000)
          - y: Y coordinate (0-1000)
        """
        session_data = self.session_manager.get_session(instance_id)
        if not session_data:
            return ToolResponse(text="Error: invalid instance_id"), -0.1, {"success": False}
        
        # Check if already stopped
        if session_data.get("stopped"):
            return ToolResponse(text="Error: segmentation already stopped"), -0.05, {"success": False}
        
        # Parse coordinates
        x = parameters.get("x")
        y = parameters.get("y")
        
        if x is None or y is None:
            return ToolResponse(text="Error: missing x or y coordinate"), -0.05, {"success": False}
        
        # Convert from 0-1000 to image pixel coordinates
        img = session_data["image"]
        img_width, img_height = img.size
        pixel_x = int(x * img_width / 1000)
        pixel_y = int(y * img_height / 1000)
        
        # Add negative point (label=0)
        point = [pixel_x, pixel_y]
        self.session_manager.update_clicks(instance_id, point, label=0)
        
        # Generate updated mask
        try:
            mask = await self._predict_with_multiturn_api(session_data)
            response_text = f"Added negative point at ({x}, {y}). Mask refined."
            return ToolResponse(image=[mask], text=response_text), 0.0, {"success": True}
        except Exception as e:
            error_msg = f"Failed to generate mask: {e}"
            logger.error(error_msg)
            return ToolResponse(text=f"Error: {error_msg}"), -0.1, {"success": False}


class StopActionTool(BaseSegmentationTool):
    """Tool for stopping the refinement process."""
    
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Stop the refinement process.
        
        No parameters required.
        """
        session_data = self.session_manager.get_session(instance_id)
        if not session_data:
            return ToolResponse(text="Error: invalid instance_id"), -0.1, {"success": False}
        
        # Mark as stopped
        session_data["stopped"] = True
        
        # Return the last generated mask if available
        last_mask = session_data.get("last_mask")
        if last_mask:
            response_text = "Refinement stopped. Final mask ready."
            return ToolResponse(image=[last_mask], text=response_text), 0.0, {"success": True, "stopped": True}
        else:
            response_text = "Refinement stopped. No mask generated yet."
            return ToolResponse(text=response_text), 0.0, {"success": True, "stopped": True}
