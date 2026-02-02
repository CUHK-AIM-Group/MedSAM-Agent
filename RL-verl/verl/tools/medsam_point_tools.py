# Copyright 2025
# MedSAM point-based refinement tools for interactive segmentation

import asyncio
import logging
import os
from typing import Any, Optional
from uuid import uuid4
from functools import wraps

import numpy as np
import requests
from PIL import Image
import io
import base64

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def retry_on_connection_error(max_retries=3, delay=1.0):
    """Decorator to retry on connection errors."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (ConnectionResetError, ConnectionError, requests.exceptions.ConnectionError, 
                        requests.exceptions.Timeout) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (attempt + 1)  # Exponential backoff
                        logger.warning(f"Connection error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Connection failed after {max_retries} attempts: {e}")
            raise last_exception
        return wrapper
    return decorator


class MedSAMPointToolBase(BaseTool):
    """Base class for MedSAM point-based tools using API server."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instances: dict[str, dict] = {}
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1.0)
        
        # API server configuration
        self.api_url = config.get("api_url", "http://localhost:8265")
        
        # Create session with connection pooling for better performance
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=100,
            pool_maxsize=100,
            max_retries=requests.adapters.Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=[500, 502, 503, 504]
            ),
            pool_block=False
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        logger.info(f"Initialized {self.__class__.__name__} with API URL: {self.api_url}, "
                   f"timeout: {self.timeout}s, max_retries: {self.max_retries}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    @retry_on_connection_error(max_retries=3, delay=1.0)
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a new segmentation session with the input image."""
        if instance_id is None:
            instance_id = str(uuid4())

        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)

        # Debug logging
        logger.debug(f"[MedSAM Tool] create called with kwargs keys: {list(kwargs.keys())}")
        logger.debug(f"[MedSAM Tool] create_kwargs keys: {list(create_kwargs.keys())}")
        
        image = kwargs.get("image")
        if image is None:
            logger.error(f"[MedSAM Tool] Missing image! kwargs: {list(kwargs.keys())}, create_kwargs: {list(create_kwargs.keys())}")
            raise ValueError("Missing required 'image' parameter in kwargs")
        
        logger.debug(f"[MedSAM Tool] Image type: {type(image)}, Image size: {image.size if hasattr(image, 'size') else 'N/A'}")

        # Convert PIL Image to bytes
        if isinstance(image, Image.Image):
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Create session via API
        files = {'image': ('image.png', img_bytes, 'image/png')}
        data = {'session_id': instance_id}
        
        # Disable proxy only for localhost API calls
        response = self.session.post(
            f"{self.api_url}/session/create",
            files=files,
            data=data,
            timeout=self.timeout,
            proxies={'http': None, 'https': None}
        )
        response.raise_for_status()
        result = response.json()
        
        # Store instance data
        self._instances[instance_id] = {
            "session_id": result["session_id"],
            "image_size": result.get("image_size"),
            "clicks": [],  # Track all clicks for this instance
            "current_mask": None,
        }
        
        logger.info(f"Created session {instance_id} with image size {result.get('image_size')}")
        return instance_id, ToolResponse()

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release resources for a specific instance."""
        if instance_id in self._instances:
            try:
                # Delete session via API (disable proxy for localhost)
                response = self.session.delete(
                    f"{self.api_url}/session/{instance_id}",
                    timeout=self.timeout,
                    proxies={'http': None, 'https': None}
                )
                response.raise_for_status()
            except Exception as e:
                logger.warning(f"Failed to delete session {instance_id}: {e}")
            finally:
                del self._instances[instance_id]

    def _decode_mask_from_base64(self, mask_base64: str) -> Image.Image:
        """Decode base64 string to PIL Image."""
        img_bytes = base64.b64decode(mask_base64)
        mask_pil = Image.open(io.BytesIO(img_bytes))
        return mask_pil


class AddPositivePointTool(MedSAMPointToolBase):
    """Tool to add a positive point to expand the segmentation mask."""

    @retry_on_connection_error(max_retries=3, delay=0.5)
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute: add a positive point and get updated mask.
        
        Expected parameters:
          - x: integer, x coordinate (0-1000)
          - y: integer, y coordinate (0-1000)
        """
        inst = self._instances.get(instance_id)
        if inst is None:
            return ToolResponse(text="Error: invalid instance_id"), -0.1, {"success": False}

        x = parameters.get("x")
        y = parameters.get("y")
        
        if x is None or y is None:
            return ToolResponse(text="Error: missing 'x' or 'y' parameter"), -0.05, {"success": False}

        # Get image size to scale coordinates from [0,1000] to actual pixels
        image_size = inst.get("image_size")
        if image_size:
            width, height = image_size
            # Scale from [0,1000] to actual image coordinates
            point_x = int(x * width / 1000)
            point_y = int(y * height / 1000)
        else:
            point_x = int(x)
            point_y = int(y)

        try:
            # Call API to add click
            data = {
                'session_id': inst["session_id"],
                'point_x': point_x,
                'point_y': point_y,
                'is_positive': True,
                'multimask_output': False
            }
            
            # Disable proxy only for localhost API calls
            response = self.session.post(
                f"{self.api_url}/session/add_click",
                data=data,
                timeout=self.timeout,
                proxies={'http': None, 'https': None}
            )
            response.raise_for_status()
            result = response.json()
            
            # Decode mask
            masks_base64 = result["masks"]
            if len(masks_base64) > 0:
                mask_pil = self._decode_mask_from_base64(masks_base64[0])
                inst["current_mask"] = mask_pil
                inst["clicks"].append((point_x, point_y, 1))  # 1 for positive
                
                response_text = f"Added positive point at ({x}, {y}). Mask updated (round {result['round_number']})."
                return ToolResponse(image=[mask_pil], text=response_text), 0.0, {"success": True}
            else:
                return ToolResponse(text="Error: no mask returned"), -0.05, {"success": False}
                
        except Exception as e:
            logger.error(f"Failed to add positive point: {e}")
            return ToolResponse(text=f"Error: {str(e)}"), -0.1, {"success": False}


class AddNegativePointTool(MedSAMPointToolBase):
    """Tool to add a negative point to refine the segmentation mask."""

    @retry_on_connection_error(max_retries=3, delay=0.5)
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute: add a negative point and get updated mask.
        
        Expected parameters:
          - x: integer, x coordinate (0-1000)
          - y: integer, y coordinate (0-1000)
        """
        inst = self._instances.get(instance_id)
        if inst is None:
            return ToolResponse(text="Error: invalid instance_id"), -0.1, {"success": False}

        x = parameters.get("x")
        y = parameters.get("y")
        
        if x is None or y is None:
            return ToolResponse(text="Error: missing 'x' or 'y' parameter"), -0.05, {"success": False}

        # Get image size to scale coordinates
        image_size = inst.get("image_size")
        if image_size:
            width, height = image_size
            point_x = int(x * width / 1000)
            point_y = int(y * height / 1000)
        else:
            point_x = int(x)
            point_y = int(y)

        try:
            # Call API to add click
            data = {
                'session_id': inst["session_id"],
                'point_x': point_x,
                'point_y': point_y,
                'is_positive': False,
                'multimask_output': False
            }
            
            # Disable proxy only for localhost API calls
            response = self.session.post(
                f"{self.api_url}/session/add_click",
                data=data,
                timeout=self.timeout,
                proxies={'http': None, 'https': None}
            )
            response.raise_for_status()
            result = response.json()
            
            # Decode mask
            masks_base64 = result["masks"]
            if len(masks_base64) > 0:
                mask_pil = self._decode_mask_from_base64(masks_base64[0])
                inst["current_mask"] = mask_pil
                inst["clicks"].append((point_x, point_y, 0))  # 0 for negative
                
                response_text = f"Added negative point at ({x}, {y}). Mask refined (round {result['round_number']})."
                return ToolResponse(image=[mask_pil], text=response_text), 0.0, {"success": True}
            else:
                return ToolResponse(text="Error: no mask returned"), -0.05, {"success": False}
                
        except Exception as e:
            logger.error(f"Failed to add negative point: {e}")
            return ToolResponse(text=f"Error: {str(e)}"), -0.1, {"success": False}


class StopActionTool(MedSAMPointToolBase):
    """Tool to stop the refinement process."""

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute: signal that refinement is complete.
        
        This tool doesn't change the mask, just returns the current mask and signals completion.
        """
        inst = self._instances.get(instance_id)
        if inst is None:
            return ToolResponse(text="Error: invalid instance_id"), -0.1, {"success": False}

        current_mask = inst.get("current_mask")
        num_clicks = len(inst.get("clicks", []))
        
        if current_mask:
            response_text = f"Refinement complete. Total clicks: {num_clicks}."
            return ToolResponse(image=[current_mask], text=response_text), 0.0, {"success": True, "stop": True}
        else:
            response_text = "No mask available. Stopping."
            return ToolResponse(text=response_text), 0.0, {"success": True, "stop": True}
