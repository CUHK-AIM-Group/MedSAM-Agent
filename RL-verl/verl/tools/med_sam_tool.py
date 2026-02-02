# Copyright 2025
# IMISNet-based segmentation tool implementation for MedSAM-Agent recipe

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import ray
import cv2
import numpy as np
import torch
from PIL import Image

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse
from qwen_vl_utils import fetch_image

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Ensure repository root is importable for shared third_party dependencies
_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


class Click:
    """Point click representation for interactive segmentation."""
    def __init__(self, is_positive: bool, coords: tuple):
        self.is_positive = is_positive
        self.coords = coords  # (y, x) format
        self.indx = None
    
    @property
    def coords_and_indx(self):
        return (*self.coords, self.indx)


def get_points_nd(clicks_list):
    """Convert clicks list to numpy arrays for SAM input."""
    points, labels = [], []
    for click in clicks_list:
        h, w = click.coords_and_indx[:2]
        points.append([w, h])  # SAM expects (x, y) format
        labels.append(int(click.is_positive))
    return np.array(points), np.array(labels)



class MedSAMTool(BaseTool):
    """IMISNet-based segmentation tool for interactive medical image segmentation.

    This tool accepts image input at create time and segmentation points at execute time.
    It uses IMISNet (an improved SAM model) to generate accurate segmentation masks.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instances: dict[str, dict] = {}
        self.timeout = config.get("timeout", 30)
        self.num_workers = config.get("num_workers", 8)
        
        # Initialize IMISNet model
        self._init_imisnet_model(config)
        
        logger.info(f"Initialized MedSAMTool with IMISNet, config={config}")

    def _init_imisnet_model(self, config: dict):
        """Initialize IMISNet segmentation model."""
        try:
            from third_party.segment_anything import sam_model_registry
            from third_party.segment_anything.predictor import IMISPredictor
            from third_party.segment_anything.model import IMISNet
            from argparse import Namespace
            
            # Setup device - IMPORTANT: Use CPU for IMISNet to avoid GPU memory conflict with VLM
            # The VLM (Qwen) will occupy most GPU memory, so we run IMISNet on CPU
            self.device = torch.device("cpu")
            logger.info("Using CPU for IMISNet model to avoid GPU memory conflict with VLM")
            
            # Model parameters
            args = Namespace()
            args.image_size = config["image_size"]
            args.sam_checkpoint = config["sam_checkpoint"]
            category_weights = config["category_weights"]
            
            # Initialize model with map_location to CPU
            sam = sam_model_registry["vit_b"](args)
            # Load checkpoint with CPU mapping
            checkpoint = torch.load(args.sam_checkpoint, map_location='cpu')
            sam.load_state_dict(checkpoint, strict=False)
            sam = sam.to(self.device)
            
            imisnet = IMISNet(sam, test_mode=True, category_weights=category_weights).to(self.device)
            self.predictor = IMISPredictor(imisnet)
            
            logger.info(f"Successfully initialized IMISNet model on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize IMISNet model: {e}")
            logger.warning("Falling back to simple circle mask generation")
            self.predictor = None

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

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
        
        # Convert PIL Image to numpy array (RGB format) for IMISNet
        if isinstance(img, Image.Image):
            img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        
        # Set image for predictor
        if self.predictor is not None:
            self.predictor.set_image(img_rgb)
        
        # Store instance data
        self._instances[instance_id] = {
            "image": img,  # PIL Image for fallback
            "image_np": img_rgb,  # RGB numpy array for IMISNet
            "clicks_list": [],  # Track all clicks
            "previous_mask": None,  # Store previous mask logits
        }

        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute segmentation given one or more points using IMISNet.

        Expected parameters:
          - points: list of [x, y] coordinates in image pixel space
          - point_labels: optional list of labels (1 for positive, 0 for negative)
                         if not provided, all points are treated as positive
          - box: optional bounding box [x1, y1, x2, y2] for additional guidance
        """
        inst = self._instances.get(instance_id)
        if inst is None:
            return ToolResponse(text="Error: invalid instance_id"), -0.1, {"success": False}

        image = inst["image"]
        image_np = inst["image_np"]
        width, height = image.size

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
        if box is not None:
            box = np.array(box)

        # Use IMISNet for prediction if available
        if self.predictor is not None:
            try:
                mask = await self._predict_with_imisnet(
                    inst, points, point_labels, box
                )
            except Exception as e:
                logger.error(f"IMISNet prediction failed: {e}, falling back to simple mask")
                mask = self._generate_simple_mask(width, height, points)
        else:
            # Fallback to simple circle mask
            mask = self._generate_simple_mask(width, height, points)

        response_text = f"Produced segmentation mask for {len(points)} point(s)."
        return ToolResponse(image=[mask], text=response_text), 0.0, {"success": True}

    async def _predict_with_imisnet(
        self, 
        inst: dict, 
        points: list, 
        point_labels: list,
        box: Optional[np.ndarray]
    ) -> Image.Image:
        """Perform segmentation using IMISNet model."""
        # Convert points to Click objects
        clicks_list = []
        for (x, y), label in zip(points, point_labels):
            # Convert (x, y) to (y, x) for Click coords
            is_positive = bool(label)
            click = Click(is_positive=is_positive, coords=(int(y), int(x)))
            clicks_list.append(click)
            inst["clicks_list"].append(click)
        
        # Convert clicks to numpy format for predictor
        points_nd, labels_nd = get_points_nd(clicks_list)
        
        # Get previous mask logits for iterative refinement
        mask_input = inst.get("previous_mask")
        
        # Predict using IMISNet
        with torch.no_grad():
            masks, logits, category_pred = self.predictor.predict(
                points_nd, 
                labels_nd, 
                mask_input=mask_input, 
                box=box
            )
        
        # Store logits for next iteration
        inst["previous_mask"] = logits
        
        # Get the best mask (first one with highest confidence)
        pred_mask = masks[0][0]  # Shape: (H, W), boolean array
        
        # Convert to PIL Image (grayscale, 0/255)
        mask_uint8 = (pred_mask.astype(np.uint8)) * 255
        mask_pil = Image.fromarray(mask_uint8, mode='L')
        
        return mask_pil

    def _generate_simple_mask(self, width: int, height: int, points: list, radius: int = 8) -> Image.Image:
        """Fallback: Generate simple circular mask around points."""
        from PIL import ImageDraw
        
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        
        for p in points:
            try:
                x, y = float(p[0]), float(p[1])
            except Exception:
                continue
            left = max(0, x - radius)
            top = max(0, y - radius)
            right = min(width, x + radius)
            bottom = min(height, y + radius)
            draw.ellipse([left, top, right, bottom], fill=255)
        
        return mask

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release resources for a specific instance."""
        if instance_id in self._instances:
            # Clear instance data
            inst = self._instances[instance_id]
            inst["clicks_list"].clear()
            inst["previous_mask"] = None
            del self._instances[instance_id]
    
    def __del__(self):
        """Cleanup when tool is destroyed."""
        if hasattr(self, 'predictor') and self.predictor is not None:
            try:
                del self.predictor
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")
