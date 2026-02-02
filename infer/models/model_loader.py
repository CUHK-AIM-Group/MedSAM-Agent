import torch
from pathlib import Path
import sys
import os
import sys
import re
import json
sys.path.append('../')
project_root = Path(__file__).resolve().parents[2]
# ensure third_party/sam2 is importable
sam2_root = project_root / "third_party" / "sam2"
if str(sam2_root) not in sys.path:
    sys.path.insert(0, str(sam2_root))
# ensure third_party/segment_anything is importable
segment_root = project_root / "third_party" / "segment_anything"
if str(segment_root) not in sys.path:
    sys.path.insert(0, str(segment_root))
# also add third_party root so imports like 'segment_anything' resolve to third_party/segment_anything
third_party_root = project_root / "third_party"
if str(third_party_root) not in sys.path:
    sys.path.insert(0, str(third_party_root))
# ensure project root is importable
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- Dependency imports ---
import os
import cv2
from utils.clicker import Clicker
import time
import numpy as np
from PIL import Image
from utils.visual_utils import (
    visualize_mask_and_point,
    overlay_points,
    overlay_boxes,
    visualize_mask_and_pointlist,
    overlay_mask,
)
from PIL import Image,ImageOps
from qwen_vl_utils import process_vision_info

from third_party.segment_anything.model import IMISNet
from third_party.segment_anything import sam_model_registry
from third_party.segment_anything.predictor import IMISPredictor

from openai import OpenAI

from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Assume you have these helper functions
def overlay_mask(image_array, mask, color=[0, 255, 0], alpha=0.5):
    """Overlay the mask on the image"""
    mask = (mask > 0)
    overlay = image_array.copy()
    overlay[mask] = image_array[mask] * (1 - alpha) + np.array(color) * alpha
    return np.clip(overlay, 0, 255).astype('uint8')

def ensure_intermediate_results_dir(output_dir=None):
    """Ensure the output directory exists"""
    if output_dir:
        # If output_dir is specified, use it
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    else:
        # If not specified, use a temp directory (compat with old code)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        intermediate_dir = os.path.join(project_root, 'intermediate_results')
        os.makedirs(intermediate_dir, exist_ok=True)
        return intermediate_dir

# Color map
color_map = {
    'green': [0, 255, 0],
    'red': [255, 0, 0],
    'blue': [0, 0, 255]
}



#####Tool Using Version #####
tools_json_v2 = [
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

system_prompt_v2 = (
    "You are a professional segmentation annotator specializing in mask creation and refinement. Your core task is to segment the USER-SPECIFIED TARGET REGION from the provided image. "
    "No preliminary mask is available—you must first create an initial mask using the tool, then iteratively refine it to achieve pixel-level accuracy. "
    "The mask will be displayed as a semi-transparent green overlay; your goal is to ensure it exactly covers the entire target region and excludes all non-target areas (e.g., background, adjacent objects).\n\n"
    "# Tools\n\n"
    "You must call one function to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    f"{chr(10).join([json.dumps(tool) for tool in tools_json_v2])}\n"
    "</tools>\n\n"
    "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>\n\n"
    "Only use the provided functions to complete your task. Do not invent or assume any other functions. Carefully consider the current mask state before each action."
)

system_prompt_gpt = (
    "You are a professional segmentation annotator specializing in mask creation and refinement. Your core task is to segment the USER-SPECIFIED TARGET REGION from the provided image. "
    "No preliminary mask is available—you must first create an initial mask using the tool, then iteratively refine it to achieve pixel-level accuracy. "
    "The mask will be displayed as a semi-transparent green overlay; your goal is to ensure it exactly covers the entire target region and excludes all non-target areas (e.g., background, adjacent objects).\n\n"
    "# Important: Coordinate System\n"
    "All coordinates must be provided as RELATIVE COORDINATES normalized to the image. The coordinate range is 0-1000, where (0, 0) represents the top-left corner and (1000, 1000) represents the bottom-right corner of the image.\n\n"
    "# Tools\n\n"
    "You must call one function to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    f"{chr(10).join([json.dumps(tool) for tool in tools_json_v2])}\n"
    "</tools>\n\n"
    "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>\n\n"
    "Only use the provided functions to complete your task. Do not invent or assume any other functions. Carefully consider the current mask state before each action."
)
         
prompt_turn_1 = "<image>The target to be segmented is: {target_description}.\n Now, please analyze the original image, then decide your first action."

prompt_later_turn = "<image>Here is the updated mask after your previous action. Based on this, what is your next action? If the mask is now accurate, you can call 'stop_action' to finish."

class GroundingModel_GPT():
    def __init__(self, args):
        # Use the provided output_dir to avoid creating an unused intermediate_results folder
        output_dir = getattr(args, 'output_dir', None)
        self.workspace = ensure_intermediate_results_dir(output_dir)
        # Prefer provided args; fallback to OPENAI_API_KEY env; base_url optional
        api_key = getattr(args, 'gpt_api_key', None) or os.environ.get('OPENAI_API_KEY')
        base_url = getattr(args, 'gpt_api_base', None)
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)
        # Store model name, default gpt-4o
        self.gpt_model = getattr(args, 'gpt_model', 'gpt-4o')
        self.use_mask_module = args.use_mask_module
        self.visualize = args.visualize
        self.args = args
    
        self.predictor = None
        
        # Conversation history management - GPT version
        self.conversation_history = []
        self.max_history_length = getattr(args, 'max_history_length', 10)  # Keep 10 turns by default
        self.current_session_id = None
        self.session_histories = {}  # History records for multiple sessions
        self.intermediate_images = {}  # Intermediate image paths per session
        self.temp_dirs = set()  # Track created temp folders

    def start_new_session(self, session_id=None):
        """
        Start a new conversation session
        """
        if session_id is None:
            session_id = f"session_{time.time()}"
        
        self.current_session_id = session_id
        if session_id not in self.session_histories:
            self.session_histories[session_id] = []
        self.conversation_history = self.session_histories[session_id]
        
        # Initialize intermediate image storage for the new session
        if session_id not in self.intermediate_images:
            self.intermediate_images[session_id] = []
        
        print(f"Started new conversation session: {session_id}")
        return session_id

    def switch_session(self, session_id):
        """
        Switch to the specified conversation session
        """
        if session_id in self.session_histories:
            self.current_session_id = session_id
            self.conversation_history = self.session_histories[session_id]
            # Ensure intermediate image storage switches to the matching session
            if session_id not in self.intermediate_images:
                self.intermediate_images[session_id] = []
            print(f"Switched to session: {session_id}")
        else:
            print(f"Session {session_id} not found, creating new session")
            self.start_new_session(session_id)

    def get_system_prompt(self):
        """
        Get the system prompt, including full task description and output format requirements (GPT version with coordinate system)
        """
        return system_prompt_gpt
        
    def build_prompt(self, init_inputs, last_ref_box_str=None, use_history=True, reset_history=False):
        """
        Build OpenAI GPT messages format, with conversation history support
        """
        if reset_history:
            self.clear_current_session_history()
            
        image_path = init_inputs['img_path']
        caption = init_inputs['caption'][0]

        # Determine if this is the first or a later turn
        is_first_turn = len(self.conversation_history) == 0
        
        # Build the user message text for the current turn
        if is_first_turn:
            # First turn uses prompt_turn_1
            current_text = prompt_turn_1.format(target_description=caption)
        else:
            # Later turns use prompt_later_turn
            action_num = len(self.conversation_history) // 2 + 1
            current_text = prompt_later_turn.format(action_num=action_num)

        if last_ref_box_str:
            current_text += f" {last_ref_box_str}"

        # Build the messages list
        messages = []
        
        # Add system message
        messages.append({
            "role": "system",
            "content": self.get_system_prompt()
        })
        
        # Add history messages if enabled
        if use_history and self.conversation_history:
            # Add history, limit length to avoid long context
            history_to_use = self.conversation_history[-(self.max_history_length * 2):]
            
            # Get intermediate images for the current session
            session_intermediate_images = self.intermediate_images.get(self.current_session_id, [])
            
            for i, hist_msg in enumerate(history_to_use):
                if hist_msg['role'] == 'user':
                    # Use the intermediate image for the corresponding history turn
                    user_turn_index = i // 2  # Two messages per turn
                    if user_turn_index < len(session_intermediate_images):
                        intermediate_image_path = session_intermediate_images[user_turn_index]
                    else:
                        intermediate_image_path = image_path
                    
                    hist_content = []
                    for content in hist_msg['content']:
                        if content['type'] == 'image_url':
                            import base64
                            try:
                                if isinstance(intermediate_image_path, str) and intermediate_image_path.startswith("data:image"):
                                    image_url = intermediate_image_path
                                elif isinstance(intermediate_image_path, Image.Image):
                                    from io import BytesIO
                                    buffer = BytesIO()
                                    intermediate_image_path.save(buffer, format='PNG')
                                    image_url = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
                                else:
                                    with open(intermediate_image_path, "rb") as f:
                                        image_data = base64.b64encode(f.read()).decode('utf-8')
                                    image_url = f"data:image/png;base64,{image_data}"

                                hist_content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_url
                                    }
                                })
                            except:
                                # If image loading fails, skip this history message
                                continue
                        else:
                            hist_content.append(content)
                    
                    if hist_content:  # Only add when content exists
                        messages.append({
                            "role": "user",
                            "content": hist_content
                        })
                elif hist_msg['role'] == 'assistant':
                    # Add assistant message directly
                    messages.append({
                        "role": "assistant",
                        "content": hist_msg['content']
                    })

        # Add current turn user message (with current masked image)
        current_message = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_path  # Replaced with masked image in generate_response
                    }
                },
                {
                    "type": "text", 
                    "text": current_text
                }
            ]
        }
        messages.append(current_message)
        
        return messages, image_path

    def generate_response(self, messages, image_path, masks=None, conv=None, save_history=True, round_num=0):
        """
        Generate responses using OpenAI GPT-4o, with conversation history support
        """
        # Load original image
        if isinstance(image_path, Image.Image):
            image = image_path.convert('RGB')
            image_name = getattr(self.args, 'current_image_name', 'image')
        else:
            image = Image.open(image_path).convert('RGB')
            image_name = os.path.basename(image_path).split('.')[0]
        if hasattr(image, 'exif'):
            image = ImageOps.exif_transpose(image)

        # If a mask exists and this is not the first turn, overlay the mask on the image
        vis_image = image
        if masks is not None and round_num > 0:
            # Resize mask to match current image size
            image_np = np.array(image)
            h, w = image_np.shape[:2]
            
            # Handle mask: may be a tensor or numpy array
            if isinstance(masks, torch.Tensor):
                mask_np = masks[0].cpu().numpy()
            elif isinstance(masks, (list, tuple)):
                mask_np = masks[0]
            else:
                mask_np = masks
            
            # Resize mask to image size
            if mask_np.shape != (h, w):
                mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
                mask_resized = mask_pil.resize((w, h), Image.NEAREST)
                mask_np = (np.array(mask_resized) > 128).astype(np.uint8)
            
            mask_img = overlay_mask(image_np, mask_np, color=color_map['green'])
            vis_image = Image.fromarray(mask_img)
            print(f"[Round {round_num}] Overlayed mask on grounding input image (size: {h}x{w})")

        save_intermediate = getattr(self.args, 'save_intermediate', False)
        tmp_path = None
        tmp_image_for_prompt = vis_image

        # Save the masked image (from round 2 onward)
        if round_num > 0 and masks is not None and save_intermediate:
            sample_output_dir = getattr(self.args, 'sample_output_dir', None)
            if sample_output_dir:
                tmp_dir_path = os.path.join(sample_output_dir, 'interactions')
            else:
                tmp_dir_path = os.path.join(self.workspace, image_name)
                self.temp_dirs.add(tmp_dir_path)  # Track created temp folder
            os.makedirs(tmp_dir_path, exist_ok=True)
            tmp_path = os.path.join(tmp_dir_path, f"round_{round_num:02d}_with_mask.png")
            vis_image.save(tmp_path)
            tmp_image_for_prompt = tmp_path
            print(f"[Round {round_num}] Saved masked image to: {tmp_path}")
        elif round_num == 0 and save_intermediate:
            # Optionally save the original image for round 1
            tmp_image_for_prompt = image_path if not isinstance(image_path, Image.Image) else vis_image
        
        # Save intermediate image path for the current round (for history)
        if self.current_session_id and round_num > 0:
            if self.current_session_id not in self.intermediate_images:
                self.intermediate_images[self.current_session_id] = []
            # Add intermediate image path for the current round
            if tmp_path:
                self.intermediate_images[self.current_session_id].append(tmp_path)
            else:
                # Save in-memory image as base64 to avoid temp files
                from io import BytesIO
                import base64
                buffer = BytesIO()
                tmp_image_for_prompt.save(buffer, format='PNG')
                image_data_inline = base64.b64encode(buffer.getvalue()).decode('utf-8')
                self.intermediate_images[self.current_session_id].append(f"data:image/png;base64,{image_data_inline}")
            # Limit intermediate image count to avoid excessive storage
            max_intermediate_images = self.max_history_length
            if len(self.intermediate_images[self.current_session_id]) > max_intermediate_images:
                # Remove the oldest intermediate image file
                old_path = self.intermediate_images[self.current_session_id].pop(0)
                self._safe_remove_intermediate(old_path, "[GPT] Removed old intermediate image")

        # Convert image to base64
        import base64
        if isinstance(tmp_image_for_prompt, str):
            with open(tmp_image_for_prompt, "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
        else:
            from io import BytesIO
            buffer = BytesIO()
            tmp_image_for_prompt.save(buffer, format='PNG')
            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Update image in current message (only the last user message)
        updated_messages = []
        for i, msg in enumerate(messages):
            if msg['role'] == 'user':
                updated_content = []
                for content in msg['content']:
                    if content['type'] == 'image_url':
                        if i == len(messages) - 1:
                            # Update only the last message (current turn) with masked image
                            updated_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}"
                                }
                            })
                        else:
                            # Keep history message image paths (handled in build_prompt)
                            updated_content.append(content)
                    else:
                        updated_content.append(content)
                updated_messages.append({
                    "role": msg['role'],
                    "content": updated_content
                })
            else:
                updated_messages.append(msg)

        try:
            # Call OpenAI API
            # Read sampling params from args; default temperature=0 for determinism
            temperature = getattr(self.args, 'temperature', 0.0)
            top_p = getattr(self.args, 'top_p', 1.0)
            
            # Get model name
            model_name = getattr(self.args, 'gpt_model', self.gpt_model)
            
            response = self.client.chat.completions.create(
                model=model_name,
                messages=updated_messages,
                max_tokens=2048,
                temperature=temperature,
                top_p=top_p if temperature > 0 else 1.0
            )
            
            response_text = response.choices[0].message.content
            print(f"\nModel output: {response_text[:200]}{'...' if len(response_text) > 200 else ''}\n")
            
        except Exception as e:
            print(f"❌ GPT API call failed: {type(e).__name__}: {e}")
            response_text = ""

        # Save conversation history
        if save_history and response_text:
            self._save_conversation_turn(messages, response_text, image_path, tmp_image_for_prompt)

        # Note: do not delete tmp_path because it is needed for later rounds
        # Older intermediate images are cleaned up above

        return response_text

    def _save_conversation_turn(self, messages, response, original_image_path, intermediate_image_path):
        """
        Save the current conversation turn to history
        """
        # Extract the current turn user message (last one)
        current_user_message = messages[-1]
        
        # Save user message (using base64-encoded intermediate image)
        user_message_to_save = {
            "role": "user",
            "content": []
        }
        
        import base64
        try:
            if isinstance(intermediate_image_path, str) and intermediate_image_path.startswith("data:image"):
                image_url = intermediate_image_path
            elif isinstance(intermediate_image_path, Image.Image):
                from io import BytesIO
                buffer = BytesIO()
                intermediate_image_path.save(buffer, format='PNG')
                image_url = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
            else:
                with open(intermediate_image_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                image_url = f"data:image/png;base64,{image_data}"
            
            for content in current_user_message['content']:
                if content['type'] == 'image_url':
                    user_message_to_save['content'].append({
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    })
                else:
                    user_message_to_save['content'].append(content)
        except Exception as e:
            print(f"Error saving conversation turn: {e}")
            return
        
        # Save assistant response
        assistant_message_to_save = {
            "role": "assistant",
            "content": response
        }
        
        # Add to current session history
        self.conversation_history.append(user_message_to_save)
        self.conversation_history.append(assistant_message_to_save)
        
        # Limit history length
        if len(self.conversation_history) > self.max_history_length * 2:
            self.conversation_history = self.conversation_history[-(self.max_history_length * 2):]
        
        # Update session storage
        if self.current_session_id:
            self.session_histories[self.current_session_id] = self.conversation_history
        
        print(f"Conversation turn saved. Current history length: {len(self.conversation_history) // 2} turns")

    def process_response(self, outputs):
        """
        Parse model output: extract tool-calling function calls (GPT version)
        New format examples:
            - <tool_call>{"name": "add_bbox", "arguments": {"bbox_2d": [100, 200, 300, 400]}}</tool_call>
            - <tool_call>{"name": "add_point", "arguments": {"point_2d": [345, 167], "point_type": "positive"}}</tool_call>
            - <tool_call>{"name": "add_point", "arguments": {"point_2d": [100, 200], "point_type": "negative"}}</tool_call>
            - <tool_call>{"name": "stop_action", "arguments": {}}</tool_call>
        Legacy format (compatible):
            - <tool_call>{"name": "add_positive_point", "arguments": {"x": 500, "y": 300}}</tool_call>
            - <tool_call>{"name": "add_negative_point", "arguments": {"x": 100, "y": 200}}</tool_call>
        
        Coordinate range: 0-999 (relative coordinate system)
        
        Returns:
            is_positive: True/False/None; None indicates stop_action
            relative_coor: (x, y) normalized coordinates or None
            should_stop: whether to stop
        """
        import json
        
        is_positive = None
        relative_coor = None
        should_stop = False
        
        # Extract content inside <tool_call> tags
        tool_call_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', outputs, re.DOTALL)
        
        if tool_call_match:
            try:
                # Get raw JSON string
                raw_json = tool_call_match.group(1)
                
                # Fix common JSON format errors: missing quotes around key names
                # Example: {"x": 500, y": 430} -> {"x": 500, "y": 430}
                # Pattern: comma or left brace followed by spaces and unquoted key
                fixed_json = re.sub(r'([,{]\s*)([a-zA-Z_]\w*)(\s*":\s*)', r'\1"\2\3', raw_json)
                
                if fixed_json != raw_json:
                    print("Fixed JSON format error:")
                    print(f"  Original: {raw_json}")
                    print(f"  Fixed: {fixed_json}")
                
                # Parse JSON
                tool_call_json = json.loads(fixed_json)
                function_name = tool_call_json.get("name", "")
                arguments = tool_call_json.get("arguments", {})
                
                print(f"Parsed function call: {function_name}, args: {arguments}")
                
                # Handle add_bbox
                if function_name == "add_bbox":
                    bbox_2d = arguments.get("bbox_2d")
                    if bbox_2d and isinstance(bbox_2d, list) and len(bbox_2d) == 4:
                        # Return bbox coords [x1, y1, x2, y2], normalized from 0-999 to 0-1
                        normalized_bbox = [coord / 999.0 for coord in bbox_2d]
                        # Use bbox center as positive point
                        center_x = (normalized_bbox[0] + normalized_bbox[2]) / 2.0
                        center_y = (normalized_bbox[1] + normalized_bbox[3]) / 2.0
                        is_positive = True
                        relative_coor = (center_x, center_y)
                        print(f"✓ Using bbox center as positive point: ({center_x:.3f}, {center_y:.3f})")
                    else:
                        print(f"Warning: Invalid bbox_2d format: {bbox_2d}")
                
                # Primary format: add_point (new format)
                elif function_name == "add_point":
                    point_type = arguments.get("point_type", "").lower()
                    point_2d = arguments.get("point_2d")
                    
                    if point_type in ["positive", "negative"] and point_2d:
                        is_positive = (point_type == "positive")
                        if isinstance(point_2d, list) and len(point_2d) >= 2:
                            x, y = point_2d[0], point_2d[1]
                            # Normalize coordinates: 0-999 to 0-1
                            relative_coor = (x / 999.0, y / 999.0)
                        else:
                            print(f"Warning: Invalid point_2d format: {point_2d}")
                    else:
                        print(f"Warning: Invalid point_type or missing point_2d in add_point: {arguments}")
                
                # Legacy format: add_positive_point
                elif function_name == "add_positive_point":
                    is_positive = True
                    x = arguments.get("x")
                    y = arguments.get("y")
                    if x is not None and y is not None:
                        # Normalize coordinates: 0-999 to 0-1
                        relative_coor = (x / 999.0, y / 999.0)
                    else:
                        print(f"Warning: Positive point missing coordinates in arguments: {arguments}")
                        
                # Legacy format: add_negative_point
                elif function_name == "add_negative_point":
                    is_positive = False
                    x = arguments.get("x")
                    y = arguments.get("y")
                    if x is not None and y is not None:
                        # Normalize coordinates: 0-999 to 0-1
                        relative_coor = (x / 999.0, y / 999.0)
                    else:
                        print(f"Warning: Negative point missing coordinates in arguments: {arguments}")
                        
                elif function_name == "stop_action":
                    should_stop = True
                    is_positive = None
                    relative_coor = None
                    
                else:
                    print(f"Warning: Unknown function name: {function_name}")
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse tool_call JSON: {e}")
        else:
            print(f"⚠️ Warning: No <tool_call> tag found in output")
        
        # Return three values: is_positive, relative_coor, should_stop
        return is_positive, relative_coor, should_stop

    def clear_current_session_history(self):
        """Clear conversation history for current session"""
        # Clean intermediate images for current session
        if self.current_session_id and self.current_session_id in self.intermediate_images:
            for img_path in self.intermediate_images[self.current_session_id]:
                self._safe_remove_intermediate(img_path, "Removed intermediate image")
            self.intermediate_images[self.current_session_id] = []
        
        self.conversation_history.clear()
        if self.current_session_id:
            self.session_histories[self.current_session_id] = []
        print("Current session history cleared")

    def clear_all_sessions(self):
        """Clear history for all sessions"""
        # Clean intermediate images for all sessions
        for session_id, img_paths in self.intermediate_images.items():
            for img_path in img_paths:
                self._safe_remove_intermediate(img_path, "Removed intermediate image")
        
        self.session_histories.clear()
        self.conversation_history.clear()
        self.intermediate_images.clear()
        self.current_session_id = None
        print("All session histories cleared")

    def get_session_summary(self):
        """Get summary info for the current session"""
        intermediate_image_count = 0
        if self.current_session_id and self.current_session_id in self.intermediate_images:
            intermediate_image_count = len(self.intermediate_images[self.current_session_id])
        
        return {
            "current_session_id": self.current_session_id,
            "current_history_length": len(self.conversation_history) // 2,
            "total_sessions": len(self.session_histories),
            "max_history_length": self.max_history_length,
            "intermediate_images_count": intermediate_image_count
        }

    def load_session_history(self, history_data, session_id=None):
        """
        Load session history
        """
        if session_id is None:
            session_id = f"loaded_session_{time.time()}"
        
        self.session_histories[session_id] = history_data
        self.switch_session(session_id)
        print(f"Loaded history into session: {session_id}")

    def _safe_remove_intermediate(self, img_path, log_prefix="Removed intermediate image"):
        if not img_path:
            return
        if isinstance(img_path, Image.Image):
            return
        if isinstance(img_path, (bytes, str, Path)):
            path_str = img_path.decode("utf-8", errors="ignore") if isinstance(img_path, bytes) else str(img_path)
            if path_str.startswith("data:"):
                return
            try:
                if os.path.exists(path_str):
                    os.remove(path_str)
                    print(f"{log_prefix}: {path_str}")
            except OSError:
                pass

    def cleanup_session_images(self, session_id=None):
        """
        Manually clean intermediate images for a specific session
        """
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id and session_id in self.intermediate_images:
            for img_path in self.intermediate_images[session_id]:
                self._safe_remove_intermediate(img_path, "Cleaned up intermediate image")
            self.intermediate_images[session_id] = []
            print(f"Cleaned up all intermediate images for session: {session_id}")

    def release_resources(self):
        """Release resources"""
        # Clean all intermediate images
        for session_id, img_paths in self.intermediate_images.items():
            for img_path in img_paths:
                self._safe_remove_intermediate(img_path, "Cleaned up intermediate image")
        
        # Clean empty temp folders
        self._cleanup_empty_temp_dirs()
        
        # GPT model doesn't need explicit release, but we can clear the client
        self.client = None
    
    def _cleanup_empty_temp_dirs(self):
        """Clean empty temp folders"""
        import shutil
        for temp_dir in list(self.temp_dirs):
            if os.path.exists(temp_dir):
                try:
                    # Try to remove empty folder or folder contents
                    if not os.listdir(temp_dir):  # Folder is empty
                        os.rmdir(temp_dir)
                        print(f"Removed empty temp directory: {temp_dir}")
                    else:
                        # If folder is not empty, remove all contents
                        shutil.rmtree(temp_dir)
                        print(f"Removed temp directory and contents: {temp_dir}")
                    self.temp_dirs.discard(temp_dir)
                except Exception as e:
                    print(f"Failed to remove temp directory {temp_dir}: {e}")

class GroundingModel_QwenVL_WithHistory():
    def __init__(self, model_path, args):
        # Use the provided output_dir to avoid creating an unused intermediate_results folder
        output_dir = getattr(args, 'output_dir', None)
        self.workspace = ensure_intermediate_results_dir(output_dir)
        self.use_mask_module = args.use_mask_module
        self.visualize = args.visualize
        self.args = args

        # Load Qwen2.5-VL model, tokenizer, and processor
        # Choose precision by config: FP16, BF16, or FP32
        use_fp16 = getattr(args, 'use_fp16', False)
        if use_fp16:
            # Check BF16 support; prefer BF16 when available, otherwise FP16
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
                print("Loading model with BF16 precision")
            else:
                dtype = torch.float16
                print("Loading model with FP16 precision")
        else:
            dtype = torch.float32
            print("Loading model with FP32 precision")
        
        if "qwen2.5" in model_path or "qwen2_5" in model_path:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="auto"
            ).eval()
        elif "Qwen3" in model_path or "qwen3" in model_path:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="auto"
            ).eval()
        
        self.processor = AutoProcessor.from_pretrained(model_path)
        # print(self.processor)
        self.predictor = None
        
        # Conversation history management
        self.conversation_history = []
        self.max_history_length = getattr(args, 'max_history_length', 10)  # Keep 10 turns by default
        self.current_session_id = None
        self.session_histories = {}  # History records for multiple sessions
        self.intermediate_images = {}  # Intermediate images per session
        self.temp_dirs = set()  # Track created temp folders

    def start_new_session(self, session_id=None):
        """
        Start a new conversation session
        """
        if session_id is None:
            session_id = f"session_{time.time()}"
        
        self.current_session_id = session_id
        if session_id not in self.session_histories:
            self.session_histories[session_id] = []
        self.conversation_history = self.session_histories[session_id]
        
        # Initialize intermediate image storage for the new session
        if session_id not in self.intermediate_images:
            self.intermediate_images[session_id] = []
        
        print(f"Started new conversation session: {session_id}")
        return session_id

    def switch_session(self, session_id):
        """
        Switch to the specified conversation session
        """
        if session_id in self.session_histories:
            self.current_session_id = session_id
            self.conversation_history = self.session_histories[session_id]
            # Ensure intermediate image storage switches to the matching session
            if session_id not in self.intermediate_images:
                self.intermediate_images[session_id] = []
            print(f"Switched to session: {session_id}")
        else:
            print(f"Session {session_id} not found, creating new session")
            self.start_new_session(session_id)

    def get_system_prompt(self):
        """
        Get the system prompt, including full task description and output format requirements
        """
        return system_prompt_v2
        
    def build_prompt(self, init_inputs, last_ref_box_str=None, use_history=True, reset_history=False):
        """
        Build Qwen2.5-VL messages format, with conversation history support
        """
        if reset_history:
            self.clear_current_session_history()
            
        image_path = init_inputs['img_path']
        caption = init_inputs['caption'][0]

        # Determine if this is the first or a later turn
        is_first_turn = len(self.conversation_history) == 0
        
        # Build the user message text for the current turn
        if is_first_turn:
            # First turn uses prompt_turn_1
            current_text = prompt_turn_1.format(target_description=caption)
        else:
            # Later turns use prompt_later_turn
            action_num = len(self.conversation_history) // 2 + 1
            current_text = prompt_later_turn.format(action_num=action_num)

        if last_ref_box_str:
            current_text += f" {last_ref_box_str}"

        # Build the messages list
        messages = []
        
        # Add system message
        messages.append({
            "role": "system",
            "content": self.get_system_prompt()
        })
        
        # Add history messages if enabled
        if use_history and self.conversation_history:
            # Add history, limit length to avoid long context
            history_to_use = self.conversation_history[-(self.max_history_length * 2):]
            
            # Get intermediate images for the current session
            session_intermediate_images = self.intermediate_images.get(self.current_session_id, [])
            
            for i, hist_msg in enumerate(history_to_use):
                if hist_msg['role'] == 'user':
                    # Use the intermediate image for the corresponding history turn
                    user_turn_index = i // 2  # Two messages per turn
                    if user_turn_index < len(session_intermediate_images):
                        # Use intermediate image for the corresponding turn
                        intermediate_image_path = session_intermediate_images[user_turn_index]
                    else:
                        # If no intermediate image, use original image path
                        intermediate_image_path = image_path
                    
                    # Build history user message using intermediate image
                    hist_content = []
                    for content in hist_msg['content']:
                        if content['type'] == 'image':
                            hist_content.append({
                                "type": "image",
                                "image": intermediate_image_path
                            })
                        else:
                            hist_content.append(content)
                    
                    messages.append({
                        "role": "user",
                        "content": hist_content
                    })
                elif hist_msg['role'] == 'assistant':
                    # Add assistant message directly
                    messages.append({
                        "role": "assistant",
                        "content": hist_msg['content']
                    })

        # Add current turn user message (with current masked image)
        current_message = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path  # Replaced with masked image in generate_response
                },
                {
                    "type": "text", 
                    "text": current_text
                }
            ]
        }
        messages.append(current_message)
        
        return messages, image_path

    def generate_response(self, messages, image_path, masks=None, conv=None, save_history=True, round_num=0):
        """
        Generate responses using the official Qwen2.5-VL inference API, with history support
        """
        # Load original image
        if isinstance(image_path, Image.Image):
            image = image_path.convert('RGB')
            image_name = getattr(self.args, 'current_image_name', 'image')
        else:
            image = Image.open(image_path).convert('RGB')
            image_name = os.path.basename(image_path).split('.')[0]
        if hasattr(image, 'exif'):
            image = ImageOps.exif_transpose(image)
        
        # If a mask exists and this is not the first turn, overlay the mask on the image
        vis_image = image
        if masks is not None and round_num > 0:
            # Resize mask to match current image size
            image_np = np.array(image)
            h, w = image_np.shape[:2]
            
            # Handle mask: may be a tensor or numpy array
            if isinstance(masks, torch.Tensor):
                mask_np = masks[0].cpu().numpy()
            elif isinstance(masks, (list, tuple)):
                mask_np = masks[0]
            else:
                mask_np = masks
            
            # Resize mask to image size
            if mask_np.shape != (h, w):
                mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
                mask_resized = mask_pil.resize((w, h), Image.NEAREST)
                mask_np = (np.array(mask_resized) > 128).astype(np.uint8)
            
            mask_img = overlay_mask(image_np, mask_np, color=color_map['green'])
            vis_image = Image.fromarray(mask_img)
            print(f"[Round {round_num}] Overlayed mask on grounding input image (size: {h}x{w})")

        save_intermediate = getattr(self.args, 'save_intermediate', False)
        tmp_path = None
        tmp_image_for_prompt = vis_image

        # Save the masked image (from round 2 onward)
        if round_num > 0 and masks is not None and save_intermediate:
            sample_output_dir = getattr(self.args, 'sample_output_dir', None)
            if sample_output_dir:
                tmp_dir_path = os.path.join(sample_output_dir, 'interactions')
            else:
                tmp_dir_path = os.path.join(self.workspace, image_name)
                self.temp_dirs.add(tmp_dir_path)  # Track created temp folder
            os.makedirs(tmp_dir_path, exist_ok=True)
            tmp_path = os.path.join(tmp_dir_path, f"round_{round_num:02d}_with_mask.png")
            vis_image.save(tmp_path)
            tmp_image_for_prompt = tmp_path
            print(f"[Round {round_num}] Saved masked image to: {tmp_path}")
        elif round_num == 0 and save_intermediate:
            tmp_image_for_prompt = image_path if not isinstance(image_path, Image.Image) else vis_image
        
        # Save intermediate image path for current round (for history)
        if self.current_session_id and round_num > 0:
            if self.current_session_id not in self.intermediate_images:
                self.intermediate_images[self.current_session_id] = []
            # Add intermediate image path for current round
            if tmp_path:
                self.intermediate_images[self.current_session_id].append(tmp_path)
            else:
                self.intermediate_images[self.current_session_id].append(tmp_image_for_prompt)
            # Limit intermediate image count to avoid excessive storage
            max_intermediate_images = self.max_history_length
            if len(self.intermediate_images[self.current_session_id]) > max_intermediate_images:
                # Remove the oldest intermediate image file
                old_path = self.intermediate_images[self.current_session_id].pop(0)
                if isinstance(old_path, str) and os.path.exists(old_path):
                    try:
                        os.remove(old_path)
                        print(f"[Qwen] Removed old intermediate image: {old_path}")
                    except:
                        pass
        # Update image in current message (only the last user message)
        updated_messages = []
        for i, msg in enumerate(messages):
            if msg['role'] == 'user':
                updated_content = []
                for content in msg['content']:
                    if content['type'] == 'image':
                        if i == len(messages) - 1:
                            # Update only the last message (current turn) with masked image
                            updated_content.append({
                                "type": "image",
                                "image": tmp_image_for_prompt  # Use masked image (path or in-memory)
                            })
                        else:
                            # Keep history message image paths (handled in build_prompt)
                            updated_content.append(content)
                    else:
                        updated_content.append(content)
                updated_messages.append({
                    "role": msg['role'],
                    "content": updated_content
                })
            else:
                updated_messages.append(msg)
                    
        # Use official inference flow
        text = self.processor.apply_chat_template(
            updated_messages, tokenize=False, add_generation_prompt=True
        )
        if "Qwen3VLProcessor" in self.processor.__class__.__name__:
            image_inputs, video_inputs = process_vision_info(updated_messages, image_patch_size=16)
        else:
            image_inputs, video_inputs = process_vision_info(updated_messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )          
        # print("Debug image len", len(image_inputs))
        # print("Debug image size and type:", image_inputs[0])
        inputs = inputs.to("cuda")

        # Generate output
        # Read sampling params from args; default deterministic (do_sample=False)
        do_sample = getattr(self.args, 'do_sample', False)
        temperature = getattr(self.args, 'temperature', 1.0) if do_sample else 1.0
        top_p = getattr(self.args, 'top_p', 1.0) if do_sample else 1.0
        top_k = getattr(self.args, 'top_k', 50) if do_sample else 50
        
        generation_config = {
            'max_new_tokens': 128,
            'do_sample': do_sample,
        }
        
        if do_sample:
            generation_config['temperature'] = temperature
            generation_config['top_p'] = top_p
            generation_config['top_k'] = top_k
        
        generated_ids = self.model.generate(**inputs, **generation_config)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = output_text[0] if output_text else ""

        # Save conversation history
        if save_history and response:
            self._save_conversation_turn(messages, response, image_path, tmp_image_for_prompt)

        # Note: do not delete tmp_path because it is needed for later rounds
        # Older intermediate images are cleaned up above

        return response

    def _save_conversation_turn(self, messages, response, original_image_path, intermediate_image_path):
        """
        Save the current conversation turn to history
        """
        # Extract the current turn user message (last one)
        current_user_message = messages[-1]
        
        # Save user message (using intermediate image path)
        user_message_to_save = {
            "role": "user",
            "content": []
        }
        
        for content in current_user_message['content']:
            if content['type'] == 'image':
                user_message_to_save['content'].append({
                    "type": "image",
                    "image": intermediate_image_path  # Use intermediate image path or in-memory image
                })
            else:
                user_message_to_save['content'].append(content)
        
        # Save assistant response
        assistant_message_to_save = {
            "role": "assistant",
            "content": response
        }
        
        # Add to current session history
        self.conversation_history.append(user_message_to_save)
        self.conversation_history.append(assistant_message_to_save)
        
        # Limit history length
        if len(self.conversation_history) > self.max_history_length * 2:
            self.conversation_history = self.conversation_history[-(self.max_history_length * 2):]
        
        # Update session storage
        if self.current_session_id:
            self.session_histories[self.current_session_id] = self.conversation_history
        
        print(f"Conversation turn saved. Current history length: {len(self.conversation_history) // 2} turns")

    def process_response(self, outputs):
        """
        Parse model output: extract tool-calling function calls (Qwen version)
        New format examples:
            - <tool_call>{"name": "add_point", "arguments": {"point_2d": [345, 167], "point_type": "positive"}}</tool_call>
            - <tool_call>{"name": "add_point", "arguments": {"point_2d": [100, 200], "point_type": "negative"}}</tool_call>
            - <tool_call>{"name": "stop_action", "arguments": {}}</tool_call>
        Legacy format (compatible):
            - <tool_call>{"name": "add_positive_point", "arguments": {"x": 500, "y": 300}}</tool_call>
            - <tool_call>{"name": "add_negative_point", "arguments": {"x": 100, "y": 200}}</tool_call>
        
        Coordinate range: 0-999 (relative coordinate system)
        
        Returns:
            is_positive: True/False/None; None indicates stop_action
            relative_coor: (x, y) normalized coordinates or None
            should_stop: whether to stop
        """
        import json
        
        is_positive = None
        relative_coor = None
        should_stop = False
        
        # Extract content inside <tool_call> tags
        tool_call_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', outputs, re.DOTALL)
        
        if tool_call_match:
            try:
                # Get raw JSON string
                raw_json = tool_call_match.group(1)
                
                # Fix common JSON format errors: missing quotes around key names
                # Example: {"x": 500, y": 430} -> {"x": 500, "y": 430}
                # Pattern: comma or left brace followed by spaces and unquoted key
                fixed_json = re.sub(r'([,{]\s*)([a-zA-Z_]\w*)(\s*":\s*)', r'\1"\2\3', raw_json)
                
                if fixed_json != raw_json:
                    print("Fixed JSON format error:")
                    print(f"  Original: {raw_json}")
                    print(f"  Fixed: {fixed_json}")
                
                # Parse JSON
                tool_call_json = json.loads(fixed_json)
                function_name = tool_call_json.get("name", "")
                arguments = tool_call_json.get("arguments", {})
                
                print(f"Parsed function call: {function_name}, args: {arguments}")
                
                # Handle add_bbox
                if function_name == "add_bbox":
                    bbox_2d = arguments.get("bbox_2d")
                    if bbox_2d and isinstance(bbox_2d, list) and len(bbox_2d) == 4:
                        # Return special 'bbox' marker and bbox coords [x1, y1, x2, y3], normalized 0-999 to 0-1
                        normalized_bbox = [coord / 999.0 for coord in bbox_2d]
                        is_positive = 'bbox'  # Special marker indicates a bbox
                        relative_coor = normalized_bbox  # Return normalized bbox coords
                    else:
                        print(f"Warning: Invalid bbox_2d format: {bbox_2d}")
                
                # Primary format: add_point (new format)
                elif function_name == "add_point":
                    point_type = arguments.get("point_type", "").lower()
                    point_2d = arguments.get("point_2d")
                    
                    if point_type in ["positive", "negative"] and point_2d:
                        is_positive = (point_type == "positive")
                        if isinstance(point_2d, list) and len(point_2d) >= 2:
                            x, y = point_2d[0], point_2d[1]
                            # Normalize coordinates: 0-999 to 0-1
                            relative_coor = (x / 999.0, y / 999.0)
                        else:
                            print(f"Warning: Invalid point_2d format: {point_2d}")
                    else:
                        print(f"Warning: Invalid point_type or missing point_2d in add_point: {arguments}")
                
                # Legacy format: add_positive_point
                elif function_name == "add_positive_point":
                    is_positive = True
                    x = arguments.get("x")
                    y = arguments.get("y")
                    if x is not None and y is not None:
                        # Normalize coordinates: 0-999 to 0-1
                        relative_coor = (x / 999.0, y / 999.0)
                    else:
                        print(f"Warning: Positive point missing coordinates in arguments: {arguments}")
                        
                # Legacy format: add_negative_point
                elif function_name == "add_negative_point":
                    is_positive = False
                    x = arguments.get("x")
                    y = arguments.get("y")
                    if x is not None and y is not None:
                        # Normalize coordinates: 0-999 to 0-1
                        relative_coor = (x / 999.0, y / 999.0)
                    else:
                        print(f"Warning: Negative point missing coordinates in arguments: {arguments}")
                        
                elif function_name == "stop_action":
                    should_stop = True
                    is_positive = None
                    relative_coor = None
                    
                else:
                    print(f"Warning: Unknown function name: {function_name}")
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse tool_call JSON: {e}")
                print(f"Raw content: {tool_call_match.group(1)}")
        else:
            print(f"Warning: No <tool_call> tag found in output: {outputs}")
        
        # Return three values: is_positive, relative_coor, should_stop
        return is_positive, relative_coor, should_stop

    def clear_current_session_history(self):
        """Clear conversation history for current session"""
        # Clean intermediate images for current session
        if self.current_session_id and self.current_session_id in self.intermediate_images:
            for img_path in self.intermediate_images[self.current_session_id]:
                if os.path.exists(img_path):
                    try:
                        os.remove(img_path)
                        print(f"Removed intermediate image: {img_path}")
                    except:
                        pass
            self.intermediate_images[self.current_session_id] = []
        
        self.conversation_history.clear()
        if self.current_session_id:
            self.session_histories[self.current_session_id] = []
        print("Current session history cleared")

    def clear_all_sessions(self):
        """Clear history for all sessions"""
        # Clean intermediate images for all sessions
        for session_id, img_paths in self.intermediate_images.items():
            for img_path in img_paths:
                if os.path.exists(img_path):
                    try:
                        os.remove(img_path)
                        print(f"Removed intermediate image: {img_path}")
                    except:
                        pass
        
        self.session_histories.clear()
        self.conversation_history.clear()
        self.intermediate_images.clear()
        self.current_session_id = None
        print("All session histories cleared")

    def get_session_summary(self):
        """Get summary info for the current session"""
        intermediate_image_count = 0
        if self.current_session_id and self.current_session_id in self.intermediate_images:
            intermediate_image_count = len(self.intermediate_images[self.current_session_id])
        
        return {
            "current_session_id": self.current_session_id,
            "current_history_length": len(self.conversation_history) // 2,
            "total_sessions": len(self.session_histories),
            "max_history_length": self.max_history_length,
            "intermediate_images_count": intermediate_image_count
        }

    def _safe_remove_intermediate(self, img_path, log_prefix="Removed intermediate image"):
        if not img_path:
            return
        if isinstance(img_path, Image.Image):
            return
        if isinstance(img_path, (bytes, str, Path)):
            path_str = img_path.decode("utf-8", errors="ignore") if isinstance(img_path, bytes) else str(img_path)
            if path_str.startswith("data:"):
                return
            try:
                if os.path.exists(path_str):
                    os.remove(path_str)
                    print(f"{log_prefix}: {path_str}")
            except OSError:
                pass

    def load_session_history(self, history_data, session_id=None):
        """
        Load session history
        """
        if session_id is None:
            session_id = f"loaded_session_{time.time()}"
        
        self.session_histories[session_id] = history_data
        self.switch_session(session_id)
        print(f"Loaded history into session: {session_id}")

    def cleanup_session_images(self, session_id=None):
        """
        Manually clean intermediate images for a specific session
        """
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id and session_id in self.intermediate_images:
            for img_path in self.intermediate_images[session_id]:
                self._safe_remove_intermediate(img_path, "Cleaned up intermediate image")
            self.intermediate_images[session_id] = []
            print(f"Cleaned up all intermediate images for session: {session_id}")
        
        # Clean empty temp folders
        self._cleanup_empty_temp_dirs()

    def release_resources(self):
        """Release resources"""
        # Clean all intermediate images
        for session_id, img_paths in self.intermediate_images.items():
            for img_path in img_paths:
                self._safe_remove_intermediate(img_path, "Cleaned up intermediate image")
        
        # Clean empty temp folders
        self._cleanup_empty_temp_dirs()
        
        del self.model
        del self.processor
        torch.cuda.empty_cache()
    
    def _cleanup_empty_temp_dirs(self):
        """Clean empty temp folders"""
        import shutil
        for temp_dir in list(self.temp_dirs):
            if os.path.exists(temp_dir):
                try:
                    # Try to remove empty folder or folder contents
                    if not os.listdir(temp_dir):  # Folder is empty
                        os.rmdir(temp_dir)
                        print(f"Removed empty temp directory: {temp_dir}")
                    else:
                        # If folder is not empty, remove all contents
                        shutil.rmtree(temp_dir)
                        print(f"Removed temp directory and contents: {temp_dir}")
                    self.temp_dirs.discard(temp_dir)
                except Exception as e:
                    print(f"Failed to remove temp directory {temp_dir}: {e}")


color_map = {
    'green': (0, 255, 0),
    'red': (255, 0, 0),
    'blue': (0, 0, 255)
}

def load_segmentation_model(args):
    """
    Loads the appropriate segmentation model based on arguments.
    """
    if args.seg_model == 'sam' or args.seg_model == 'medsam':
        segmentation_model = SAMModel(args)
    elif args.seg_model == 'imisnet':

        from argparse import Namespace


        # Keep parent args device index if provided, otherwise use 0
        parent_device_idx = getattr(args, 'device', 0)

        local_args = Namespace()
        device = torch.device(f'cuda:{parent_device_idx}' if torch.cuda.is_available() else 'cpu')
        # Set to 1024 resolution to match projects like Seg-r1
        local_args.image_size = getattr(args, 'seg_image_size', 1024)
        # Prefer seg_checkpoint for unified CLI
        local_args.seg_checkpoint = getattr(args, 'seg_checkpoint', None)
        category_weights = None

        sam = sam_model_registry["vit_b"](local_args).to(device)
        imisnet = IMISNet(sam, test_mode=True, category_weights=category_weights).to(device)
        predictor = IMISPredictor(imisnet)
        segmentation_model = IMISModel(predictor)
        
    else:
        raise ValueError(f"Unknown segmentation model: {args.seg_model}")

    return segmentation_model


def load_grounding_model(args):
    """
    Loads the appropriate grounding model based on arguments.
    """

    if 'qwen' in args.grounding_model:
        grounding_model = GroundingModel_QwenVL_WithHistory(args.model, args)
    elif 'gpt' in args.grounding_model.lower():
        grounding_model = GroundingModel_GPT(args)
    else:
        raise ValueError(f"Unknown grounding model: {args.grounding_model}")

    return grounding_model

def load_model(args):
    segmentation_model = load_segmentation_model(args)
    grounding_model = load_grounding_model(args)
    return segmentation_model, grounding_model

class SegmentationModel:
    def __init__(self, predictor):
        self.predictor = predictor

    def set_input_image(self, image):
        if self.predictor is not None:
            self.predictor.set_input_image(image)

    def get_prediction(self, clicker, box=None, mask=None):
        pred_mask = self.predictor.get_prediction(clicker)
        return pred_mask > 0.49

    def image_process(self,img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def release_resources(self):
        del self.predictor
        torch.cuda.empty_cache()
        self.predictor = None

    def predict_clicks_from_mask(self,target_mask, file_name=None, object_id=None, click_num=2, pred_thr=0.49):
        """
        This function is only used for SimpleClick,
        which is used for predicting initial clicks from mask generated by gtbox and SAM.
        This is an inverse process.
        """
        predictor = self.predictor
        pred_mask = np.zeros_like(target_mask)
        clicker = Clicker(gt_mask=target_mask)
        for _ in range(click_num):
            clicker.object_id = object_id
            clicker.make_next_click(pred_mask, file_name) 
            pred_probs = predictor.get_prediction(clicker)
            pred_mask = pred_probs > pred_thr 

        return clicker.get_clicks(), pred_mask


def get_points_nd(clicks_list):

    points, labels = [], []
    for click in clicks_list:
        h, w = click.coords_and_indx[:2]
        points.append([w, h])
        labels.append(int(click.is_positive))
    return np.array(points), np.array(labels)


class IMISModel:
    def __init__(self, predictor):
        self.predictor = predictor
        self.target_size = 1024  # Segmentation model uses fixed 1024 resolution

    def set_input_image(self, image):
        if self.predictor is not None:
            self.predictor.set_image(image)

    def get_prediction(self, clicker=None, box=None, mask=None):
        pred_logits = mask
        if clicker is not None:
            if box is not None:
                box = np.array(box)
            # Interactive segmentation strategy:
            # - If mask_input exists (not first turn), only use latest points for refinement
            # - If no mask_input (first turn), use all points
            if pred_logits is not None and len(clicker.get_clicks()) > 0:
                # Not first turn: use latest points + previous mask
                clicks_list = clicker.get_last_click()
                print(f"[Iterative refinement] Using latest points + previous mask, points: {len(clicks_list)}")
            else:
                # First turn: use all points (usually just one)
                clicks_list = clicker.get_clicks()
                print(f"[First round] Using all points, points: {len(clicks_list)}")
            
            points_nd, labels_nd = get_points_nd(clicks_list)
            # preds, scores, pred_logits = 
            masks, logits, category_pred = self.predictor.predict(
                    points_nd, labels_nd, mask_input=pred_logits, box=box)
            return (masks[0][0], logits)
            # max_score_idx = np.argmax(scores)
            # pred_mask = preds[max_score_idx] > self.pred_thres
            # pred_logits = pred_logits[[max_score_idx]]
            # return (pred_mask, pred_logits)
        if box is not None:
            # box (np.ndarray or None): A length 4 array given a box prompt to the
            # model, in XYXY format.
            box = np.array(box)
            masks, logits, category_pred = self.predictor.predict(
                    None, None, mask_input=pred_logits, box=box)
            return (masks[0][0], logits)
            # preds, scores, pred_logits = self.predictor.predict(
            #     box=box, mask_input=pred_logits)
            # max_score_idx = np.argmax(scores)
            # pred_mask = preds[max_score_idx] > self.pred_thres
            # pred_logits = pred_logits[[max_score_idx]]

        # If neither clicker nor box is provided, return empty result
        raise ValueError("Either clicker or box must be provided for prediction")

    def image_process(self, img_path):
        """
        Load image and resize to 1024x1024 (keep aspect ratio, pad to square)
        """
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to 1024x1024 (match grounding_resize logic, keep aspect ratio)
        h, w = image.shape[:2]
        image_pil = Image.fromarray(image)
        
        # Use thumbnail to keep aspect ratio
        image_pil.thumbnail((self.target_size, self.target_size), Image.Resampling.LANCZOS)
        
        # Create a 1024x1024 white background
        resized_image = Image.new('RGB', (self.target_size, self.target_size), (255, 255, 255))
        
        # Paste centered
        offset = ((self.target_size - image_pil.size[0]) // 2, 
                  (self.target_size - image_pil.size[1]) // 2)
        resized_image.paste(image_pil, offset)
        
        # Convert back to numpy array
        image_resized = np.array(resized_image)
        
        print(f"Segmentation model input: resized from {w}x{h} to {self.target_size}x{self.target_size}")
        return image_resized

    def release_resources(self):
        del self.predictor
        torch.cuda.empty_cache()
        self.predictor = None



class SAMModel:
    def __init__(self, args=None):
        # Get SAM model path and config from args; supports SAM2 and MedSAM2
        seg_path = getattr(args, 'seg_checkpoint', None)
        # Hydra config path is relative to the sam2 package configs directory
        model_config = getattr(args, 'sam_config', None)
        if not model_config and args is not None:
            seg_model = getattr(args, 'seg_model', 'sam')
            if isinstance(seg_model, str) and seg_model.lower() == 'medsam':
                model_config = "configs/sam2.1/sam2.1_hiera_t.yaml"
            else:
                model_config = 'configs/sam2.1/sam2.1_hiera_b+.yaml'
        if not seg_path or not model_config:
            raise ValueError("seg_checkpoint must be provided via args for SAM/MedSAM")
        
        self.predictor = SAM2ImagePredictor(build_sam2(model_config, seg_path))
        
        # Identify model type for logging
        model_name = "SAM2" if "facebook" in seg_path else "MedSAM2" if "MedSAM" in seg_path else "SAM2"
        print(f"{model_name} model loaded: {seg_path}")

        self.pred_thres = 0.50

    def set_input_image(self, image):
        if self.predictor is not None:
            self.predictor.set_image(image)

    def get_prediction(self, clicker=None, box=None, mask=None):
        pred_logits = mask
        if clicker is not None:
            if box is not None:
                box = np.array(box)
            # Interactive segmentation strategy:
            # - If mask_input exists (not first turn), only use latest points for refinement
            # - If no mask_input (first turn), use all points
            if pred_logits is not None and len(clicker.get_clicks()) > 0:
                # Not first turn: use latest points + previous mask
                clicks_list = clicker.get_last_click()
                print(f"[Iterative refinement] Using latest points + previous mask, points: {len(clicks_list)}")
            else:
                # First turn: use all points (usually just one)
                clicks_list = clicker.get_clicks()
                print(f"[First round] Using all points, points: {len(clicks_list)}")
            
            points_nd, labels_nd = get_points_nd(clicks_list)
            masks, scores, logits = self.predictor.predict(
                    point_coords=points_nd, point_labels=labels_nd, mask_input=pred_logits, box=box)
            # SAM2 returns CxHxW; choose the highest-score mask
            # masks: [C, H, W], logits: [C, 256, 256]
            max_score_idx = np.argmax(scores)
            return (masks[max_score_idx], logits[[max_score_idx]])  # Keep logits shape [1, 256, 256]
        if box is not None:
            # box (np.ndarray or None): A length 4 array given a box prompt to the
            # model, in XYXY format.
            box = np.array(box)
            masks, scores, logits = self.predictor.predict(
                box=box, mask_input=pred_logits)
            # SAM2 returns CxHxW; choose the highest-score mask
            max_score_idx = np.argmax(scores)
            return (masks[max_score_idx], logits[[max_score_idx]])  # Keep logits shape [1, 256, 256]

        # If neither clicker nor box is provided, return empty result
        raise ValueError("Either clicker or box must be provided for prediction")

    def image_process(self,img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def release_resources(self):
        del self.predictor
        torch.cuda.empty_cache()
        self.predictor = None

