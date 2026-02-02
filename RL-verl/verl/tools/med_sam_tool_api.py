# Copyright 2025
# API-based segmentation tool implementation for MedSAM-Agent recipe

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


class MedSAMToolAPI(BaseTool):
    """API-based segmentation tool for interactive medical image segmentation.

    This tool accepts image input at create time and segmentation points at execute time.
    It calls an external IMISNet API server to generate accurate segmentation masks.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instances: dict[str, dict] = {}
        self.timeout = config.get("timeout", 30)
        self.num_workers = config.get("num_workers", 8)
        
        # API configuration
        self.api_base_url = config.get("api_base_url", "http://localhost:8000")
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1.0)
        
        # Multi-turn interaction configuration
        self.use_multiturn = config.get("use_multiturn", True)  # Enable multi-turn by default
        
        # Initialize requests session - keep proxy for external requests
        self.session = requests.Session()
        
        logger.info(f"Initialized MedSAMToolAPI with API server at {self.api_base_url}, "
                   f"multi-turn mode: {self.use_multiturn}")

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
        
        # Convert PIL Image to bytes
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
                
                return session_id
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} to create session failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a new segmentation instance with the input image."""
        if instance_id is None:
            instance_id = str(uuid4())

        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)

        image = kwargs.get("image")
        if image is None:
            raise ValueError("Missing required 'image' parameter in kwargs")

        # Fetch and process image
        img = fetch_image({"image": image})
        
        # Initialize instance data
        self._instances[instance_id] = {
            "image": img,  # PIL Image for API calls
            "clicks_list": [],  # Track all clicks
            "api_session_id": None,  # API session ID for multi-turn mode
        }
        
        # If multi-turn mode, create API session
        if self.use_multiturn:
            try:
                api_session_id = await self._create_api_session(img)
                self._instances[instance_id]["api_session_id"] = api_session_id
                logger.info(f"Created API session {api_session_id} for instance {instance_id}")
            except Exception as e:
                logger.warning(f"Failed to create API session, falling back to single-shot mode: {e}")
                self._instances[instance_id]["api_session_id"] = None

        logger.info(f"Created instance {instance_id} with image size {img.size}")
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute segmentation given one or more points using IMISNet API.

        Expected parameters:
          - points: list of [x, y] coordinates in image pixel space
          - point_labels: optional list of labels (1 for positive, 0 for negative)
                         if not provided, all points are treated as positive
          - box: optional bounding box [x1, y1, x2, y2] for additional guidance
        """
        inst = self._instances.get(instance_id)
        if inst is None:
            return ToolResponse(text="Error: invalid instance_id"), -0.1, {"success": False}

        # Parse points parameter
        points = parameters.get("points") or parameters.get("point")
        if not points:
            return ToolResponse(text="Error: missing 'points' parameter"), -0.05, {"success": False}

        # Normalize single point to list
        if isinstance(points[0], (int, float)):
            points = [points]

        # Parse point labels (default all positive)
        point_labels = parameters.get("point_labels")
        if point_labels is None:
            point_labels = [1] * len(points)  # All positive by default
        
        # Parse optional box
        box = parameters.get("box")

        # Track clicks for multi-turn mode
        for point, label in zip(points, point_labels):
            inst["clicks_list"].append((point, label))

        # Call API for prediction
        try:
            # Use multi-turn API if session exists
            if inst.get("api_session_id"):
                mask = await self._predict_with_multiturn_api(
                    inst, points, point_labels, box
                )
            else:
                # Fall back to single-shot API
                mask = await self._predict_with_api(
                    inst, points, point_labels, box
                )
            response_text = f"Produced segmentation mask for {len(points)} point(s)."
            return ToolResponse(image=[mask], text=response_text), 0.0, {"success": True}
        except Exception as e:
            error_msg = f"API prediction failed: {e}"
            logger.error(error_msg)
            return ToolResponse(text=f"Error: {error_msg}"), -0.1, {"success": False}

    async def _predict_with_api(
        self, 
        inst: dict, 
        points: list, 
        point_labels: list,
        box: Optional[list]
    ) -> Image.Image:
        """Perform segmentation using IMISNet API server."""
        import json
        import asyncio
        
        image_pil = inst["image"]
        
        # Convert PIL Image to bytes for multipart upload
        img_buffer = BytesIO()
        image_pil.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        
        # Prepare multipart form data
        # API expects: point_coords (JSON string), point_labels (JSON string), bbox (JSON string)
        files = {
            'image': ('image.png', img_buffer, 'image/png')
        }
        
        data = {
            'point_coords': json.dumps(points),
            'point_labels': json.dumps(point_labels),
            'multimask_output': 'false'
        }
        
        if box is not None:
            data['bbox'] = json.dumps(box)
        
        # Call API with retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Reset buffer position for retries
                img_buffer.seek(0)
                
                # Disable proxy only for localhost API calls
                response = self.session.post(
                    f"{self.api_base_url}/predict",
                    files=files,
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
                
                # Parse response
                result = response.json()
                
                # API returns: {"masks": [...], "category_pred": [...], "message": "..."}
                masks_base64 = result.get("masks", [])
                if not masks_base64:
                    error_msg = "API response missing 'masks' field or masks list is empty"
                    logger.warning(error_msg)
                    last_error = error_msg
                    
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay)
                    continue
                
                # Use the first mask (highest confidence)
                mask_base64 = masks_base64[0]
                mask_pil = self._decode_base64_to_image(mask_base64)
                
                logger.debug(f"Successfully got mask from API: {mask_pil.size}, mode={mask_pil.mode}")
                return mask_pil
                
            except requests.exceptions.RequestException as e:
                error_msg = f"Network error: {e}"
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries}: {error_msg}")
                last_error = error_msg
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                continue
            except Exception as e:
                error_msg = f"Unexpected error: {e}"
                logger.error(error_msg)
                last_error = error_msg
                break
        
        # All retries failed
        raise RuntimeError(f"Failed to get prediction from API after {self.max_retries} attempts. Last error: {last_error}")

    async def _predict_with_multiturn_api(
        self, 
        inst: dict, 
        points: list, 
        point_labels: list,
        box: Optional[list]
    ) -> Image.Image:
        """Perform segmentation using multi-turn session API.
        
        This method uses the cumulative approach where each new click is added
        to the session, and the API uses previous_mask for iterative refinement.
        """
        import json
        import asyncio
        
        session_id = inst["api_session_id"]
        
        # Use session/segment endpoint for precise control with all accumulated clicks
        # This allows us to pass all clicks at once while still leveraging previous mask
        all_points = [point for point, _ in inst["clicks_list"]]
        all_labels = [label for _, label in inst["clicks_list"]]
        
        data = {
            'session_id': session_id,
            'point_coords': json.dumps(all_points),
            'point_labels': json.dumps(all_labels),
            'use_previous_mask': True  # Enable iterative refinement
        }
        
        if box is not None:
            data['bbox'] = json.dumps(box)
        
        # Call API with retry logic
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
                
                # Parse response
                result = response.json()
                
                masks_base64 = result.get("masks", [])
                if not masks_base64:
                    error_msg = "API response missing 'masks' field or masks list is empty"
                    logger.warning(error_msg)
                    last_error = error_msg
                    
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay)
                    continue
                
                # Use the first mask (highest confidence)
                mask_base64 = masks_base64[0]
                mask_pil = self._decode_base64_to_image(mask_base64)
                
                logger.debug(f"Successfully got mask from multi-turn API: {mask_pil.size}, "
                           f"round={result.get('round_number')}, clicks={len(all_points)}")
                return mask_pil
                
            except requests.exceptions.RequestException as e:
                error_msg = f"Network error: {e}"
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries}: {error_msg}")
                last_error = error_msg
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                continue
            except Exception as e:
                error_msg = f"Unexpected error: {e}"
                logger.error(error_msg)
                last_error = error_msg
                break
        
        # All retries failed
        raise RuntimeError(f"Failed to get prediction from multi-turn API after {self.max_retries} attempts. Last error: {last_error}")

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release resources for a specific instance."""
        if instance_id in self._instances:
            inst = self._instances[instance_id]
            
            # Clean up API session if exists
            api_session_id = inst.get("api_session_id")
            if api_session_id:
                try:
                    response = self.session.delete(
                        f"{self.api_base_url}/session/{api_session_id}",
                        timeout=self.timeout,
                        proxies=self.proxies
                    )
                    if response.status_code == 200:
                        logger.debug(f"Deleted API session {api_session_id}")
                    else:
                        logger.warning(f"Failed to delete API session {api_session_id}: {response.status_code}")
                except Exception as e:
                    logger.warning(f"Error deleting API session {api_session_id}: {e}")
            
            # Clear instance data
            inst["clicks_list"].clear()
            del self._instances[instance_id]
            logger.debug(f"Released instance {instance_id}")
