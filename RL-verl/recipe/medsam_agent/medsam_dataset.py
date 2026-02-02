# Copyright 2025
# MedSAM-Agent Dataset implementation

import io
import logging
import numpy as np
from PIL import Image
import torch
import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)


class MedSAMDataset(RLHFDataset):
    """
    Custom dataset for MedSAM-Agent project.
    
    Loads medical images and ground truth segmentation masks from parquet files.
    Columns expected: 'prompt', 'images', 'reward_model', 'extra_info'
    
    Note: This class inherits from RLHFDataset and only overrides __getitem__.
    The __init__ method is inherited from the parent class.
    
    Args:
        qwen_image_size: Resolution to resize images for Qwen model input (default: 512)
                        Original 1024 resolution is preserved for segmentation tools
    """
    
    def __init__(self, *args, qwen_image_size=512, **kwargs):
        super().__init__(*args, **kwargs)
        self.qwen_image_size = qwen_image_size
    
    def __getitem__(self, item):
        """
        Returns a dictionary with the following keys:
        - raw_prompt: messages for chat template
        - multi_modal_data: {"image": [PIL.Image]}
        - tools_kwargs: {"med_sam_tool": {"create_kwargs": {"image": PIL.Image}}}
        - agent_name: "tool_agent"
        - ground_truth_mask: PIL.Image (for reward computation)
        - input_ids, attention_mask, position_ids (for training)
        """
        # Set keys for this dataset (since we don't override __init__)
        image_key = 'images'
        reward_model_key = 'reward_model'
        
        row_dict: dict = self.dataframe[item]
        
        # Extract messages from prompt (already in chat format as numpy array)
        prompt_data = row_dict['extra_info']['target_description']
        if isinstance(prompt_data, np.ndarray):
            # Convert numpy array to list
            messages = prompt_data.tolist()
        elif isinstance(prompt_data, list):
            messages = prompt_data
        else:
            # Fallback: create messages for iterative refinement
            # Get target description from prompt_data or use default
            target_description = str(prompt_data) if prompt_data else "the object in the image"
            
            # Build system prompt inline to avoid circular import
            import json
            
            #####Tool Using Version #####
            tools_json = [
                {
                    "type": "function",
                    "function": {
                        "name": "add_bbox",
                        "description": "Add a bounding box to initialize or refine the segmentation.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "bbox_2d": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "minItems": 4,
                                    "maxItems": 4,
                                    "description": "2D bounding box in [x1, y1, x2, y2] format"
                                }
                            },
                            "required": ["bbox_2d"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "add_point",
                        "description": "Add a point to refine the mask (positive to include areas, negative to exclude areas).",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "point_2d": {
                                    "type": "array",
                                    "items": {
                                        "type": "integer"
                                    },
                                    "minItems": 2,
                                    "maxItems": 2,
                                    "description": "2D coordinate point in [x, y] format, with x and y in range [0, 999]"
                                },
                                "point_type": {
                                    "type": "string",
                                    "enum": ["positive", "negative"],
                                    "description": "Type of point: 'positive' to expand mask, 'negative' to refine mask"
                                }
                            },
                            "required": ["point_2d", "point_type"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "stop_action",
                        "description": "Stop the refinement process when the mask accurately covers the target object.",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                }
            ]

            system_prompt = (
                "You are a professional segmentation annotator specializing in mask creation and refinement. Your core task is to segment the USER-SPECIFIED TARGET REGION from the provided image. "
                "No preliminary mask is available—you must first create an initial mask using the tool, then iteratively refine it to achieve pixel-level accuracy. "
                "The mask will be displayed as a semi-transparent green overlay; your goal is to ensure it exactly covers the entire target region and excludes all non-target areas (e.g., background, adjacent objects).\n\n"
                "# Tools\n\n"
                "You must call one function to assist with the user query.\n\n"
                "You are provided with function signatures within <tools></tools> XML tags:\n"
                "<tools>\n"
                f"{chr(10).join([json.dumps(tool) for tool in tools_json])}\n"
                "</tools>\n\n"
                "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
                "<tool_call>\n"
                '{"name": <function-name>, "arguments": <args-json-object>}\n'
                "</tool_call>\n\n"
                "Only use the provided functions to complete your task. Do not invent or assume any other functions. Carefully consider the current mask state before each action."
            )

            # First, load the images from parquet data
            # Keep both original (1024) and resized (qwen_image_size) versions
            images = None  # For Qwen model (resized)
            original_images = None  # For segmentation tools (1024)
            images_data = row_dict.get(image_key)
            if images_data is not None:
                if isinstance(images_data, np.ndarray):
                    images_data = images_data.tolist()
                
                if isinstance(images_data, list) and len(images_data) > 0:
                    images = []
                    original_images = []
                    for img_item in images_data:
                        if isinstance(img_item, dict) and "bytes" in img_item:
                            try:
                                # Load original image
                                original_img = Image.open(io.BytesIO(img_item["bytes"])).convert('RGB')
                                original_images.append(original_img)
                                
                                # Create resized version for Qwen model
                                resized_img = original_img.resize(
                                    (self.qwen_image_size, self.qwen_image_size),
                                    Image.BILINEAR
                                )
                                images.append(resized_img)
                            except Exception as e:
                                logger.error(f"Failed to load image from bytes: {e}")
                        elif isinstance(img_item, dict) and "path" in img_item:
                            logger.warning(f"Image path provided but not loading: {img_item['path']}")
                        else:
                            logger.warning(f"Unexpected image item format: {type(img_item)}, keys: {img_item.keys() if isinstance(img_item, dict) else 'N/A'}")
            
            if not images:
                logger.warning(f"No images loaded for item {item}, images_data type: {type(images_data)}, len: {len(images_data) if isinstance(images_data, (list, tuple)) else 'N/A'}")
                images = []
                original_images = []
            
            # Build messages with resized image for Qwen model
            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": images[0] if images else None,  # Resized image for Qwen
                        },
                        {
                            "type": "text",
                            "text": f"The target to be segmented is: {target_description}.\nNow, please analyze the original image, then decide your first action."
                        }
                    ],
                },
            ]
        
        row_dict[self.prompt_key] = messages
        model_inputs = {}
        
        # Store images for multi_modal_data
        multi_modal_data = {}
        if images:
            multi_modal_data["image"] = images
            logger.debug(f"Loaded {len(images)} images for item {item}")
        
        # Process ground truth mask from reward_model
        gt_mask = None
        reward_model_data = row_dict.get(reward_model_key)
        if reward_model_data and isinstance(reward_model_data, dict):
            gt_data = reward_model_data.get('ground_truth')
            if gt_data:
                if isinstance(gt_data, dict) and "bytes" in gt_data:
                    gt_mask = Image.open(io.BytesIO(gt_data["bytes"]))
                elif isinstance(gt_data, bytes):
                    gt_mask = Image.open(io.BytesIO(gt_data))
                else:
                    # Assume it's already a PIL Image or numpy array
                    gt_mask = gt_data
        
        # Use processor if available
        if self.processor is not None:
            # Apply chat template to get text prompt
            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            
            # Extract images and videos from messages using qwen_vl_utils
            # This handles the structured format with PIL Images embedded
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
            else:
                image_inputs, video_inputs = process_vision_info(messages)
            
            # Use do_resize=False to avoid duplicate resizing
            model_inputs = self.processor(
                text=[raw_prompt],
                images=image_inputs, 
                videos=video_inputs,
                # do_resize=False,
                padding=True,
                return_tensors="pt"
            )
            
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
            
            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")
            
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)
        else:
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
        
        # Always set multi_modal_data regardless of processor availability
        row_dict["multi_modal_data"] = multi_modal_data
        
        # Postprocess input tensors
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        
        # Compute position_ids
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            is_qwen3vl = "Qwen3VLProcessor" in self.processor.__class__.__name__
            if is_qwen3vl:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index
            
            # Qwen3VL uses different get_rope_index signature (no second_per_grid_ts parameter)
            if is_qwen3vl:
                vision_position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    attention_mask=attention_mask[0],
                )  # (3, seq_length)
            else:
                vision_position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )  # (3, seq_length)
            # position_ids = [vision_position_ids]  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)
        
        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]
        
        # Encode raw_prompt_ids
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")
        
        row_dict["raw_prompt_ids"] = raw_prompt_ids
        
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
        
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt
        
        # Preserve extra_info from parquet if exists
        extra_info_data = row_dict.get("extra_info", {})
        if not isinstance(extra_info_data, dict):
            extra_info_data = {}
        
        # Add index (use 'id' from extra_info if available, otherwise use item index)
        index = extra_info_data.get("id", item)
        row_dict["index"] = index
        
        # Set up tools_kwargs for all three tools
        # All tools (add_bbox, add_point, stop_action) share the same session
        # Use original 1024 resolution images for segmentation tools
        if original_images:
            tools_kwargs = {
                "add_bbox": {
                    "create_kwargs": {"image": original_images[0]},
                },
                "add_point": {
                    "create_kwargs": {"image": original_images[0]},
                },
                "stop_action": {
                    "create_kwargs": {"image": original_images[0]},
                }
            }
            row_dict["tools_kwargs"] = tools_kwargs
        
        # Set agent_name to use MedSAMIterativeAgentLoop
        row_dict["agent_name"] = "medsam_iterative_agent"
        
        # Store ground truth mask in extra_info for reward computation
        row_dict["extra_info"] = extra_info_data
        if gt_mask is not None:
            row_dict["extra_info"]["ground_truth_mask"] = gt_mask
        
        return row_dict
