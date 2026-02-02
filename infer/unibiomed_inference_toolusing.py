import json
import os
import cv2
import numpy as np
import torch

from tqdm import tqdm
from PIL import Image

from utils.visual_utils import (
    visualize_mask_and_pointlist,
    overlay_mask,
)
from utils.clicker import Click, Clicker


def get_metrics(prediction, target):
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()

    dice = 2.0 * intersection / (prediction.sum() + target.sum())
    IoU = intersection / union

    return dice, IoU

def get_mean(save_metrics):
    mean_metrics = {}
    classes_list = []

    # first get class names
    for name in save_metrics.keys():
        each_item = save_metrics[name]
        if 'exp' in each_item.keys():
            class_name = each_item['exp']
            if class_name not in classes_list:
                classes_list.append(class_name)

    all_class_dice = []
    all_class_iou = []

    all_dice = []
    all_iou = []

    for per_class_name in classes_list:
        per_class_metrics = {}

        dice_per_class_list = []
        iou_per_class_list = []

        for name in save_metrics.keys():
            each_item = save_metrics[name]
            if 'exp' in each_item.keys():
                class_name = each_item['exp']

                if class_name == per_class_name:
                    dice_per_class_list.append(each_item['dice'])
                    iou_per_class_list.append(each_item['IoU'])

                all_dice.append(each_item['dice']), all_iou.append(each_item['IoU'])

        dice_per_class = sum(dice_per_class_list) / (len(dice_per_class_list)+1e-6)
        iou_per_class = sum(iou_per_class_list) / (len(iou_per_class_list) + 1e-6)

        print(per_class_name, dice_per_class)
        per_class_metrics['dice'] = dice_per_class
        per_class_metrics['IoU'] = iou_per_class

        all_class_dice.append(dice_per_class)
        all_class_iou.append(iou_per_class)

        mean_metrics[per_class_name] = per_class_metrics

    mean_class_dice = sum(all_class_dice) / (len(all_class_dice)+1e-6)
    mean_class_iou = sum(all_class_iou) / (len(all_class_iou) + 1e-6)
    mean_metrics['mean_class_dice'] = mean_class_dice
    mean_metrics['mean_class_iou'] = mean_class_iou

    mean_dice = sum(all_dice) / (len(all_dice) + 1e-6)
    mean_iou = sum(all_iou) / (len(all_iou) + 1e-6)
    mean_metrics['mean_dice'] = mean_dice
    mean_metrics['mean_iou'] = mean_iou

    print('mean_class_dice, mean_class_iou:', mean_class_dice, mean_class_iou)
    print('mean_dice, mean_iou:', mean_dice, mean_iou)

    return mean_metrics


def save_metrics_json(data_root, results_root, val_json_file='train.json'):

    save_json_file = os.path.join(results_root, 'results.json')

    json_file = os.path.join(data_root, val_json_file)
    with open(json_file, 'r') as file:
        data = json.load(file)

    annotations = data['annotations']

    save_metrics = {}

    for item in annotations:
        item_metric = {}

        mask_name = item['mask_file']

        mask = os.path.join(data_root, item['split']+'_mask', mask_name)
        pred = os.path.join(results_root, item['split'], mask_name)

        mask, pred = Image.open(mask), Image.open(pred)
        mask, pred = np.asarray(mask), np.asarray(pred)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        if mask.shape != pred.shape:
            mask = cv2.resize(mask, (pred.shape[1], pred.shape[0]))

        mask, pred = (mask > 0).astype(np.uint8), (pred > 0).astype(np.uint8)

        if mask.sum() > 0:
            dice, IoU = get_metrics(pred, mask)
            print('mask_name, dice, IoU:', mask_name, dice, IoU)

            item_metric['dice'] = dice
            item_metric['IoU'] = IoU
            item_metric['exp'] = item['sentences'][0]['sent']

        save_metrics[mask_name] = item_metric
        print(item_metric)
        print('\n')

    mean_metrics = get_mean(save_metrics)

    # new: want mean metrics in the first line
    new_save_metrics = {}
    new_save_metrics['mean'] = mean_metrics

    # new: want mean metrics in the first line
    for key in save_metrics.keys():
        new_save_metrics[key] = save_metrics[key]

    with open(save_json_file, 'w', encoding='utf-8') as json_file:
        json.dump(new_save_metrics, json_file, ensure_ascii=False, indent=4)



class InferenceArgs:
    """
    Inference configuration class
    Contains all parameters needed during inference
    """
    def __init__(self):
        # Basic parameters
        self.n_clicks = 5                    # Maximum number of clicks
        self.visualize = True                # Whether to save visualizations
        # Resize resolution for grounding model (only passed to grounding model),
        # if set to None then no resize is applied to grounding
        self.grounding_resize = 512         # Grounding model input resolution (e.g., 512)

        # Input resolution used/expected internally by segmentation model (only passed to model loader),
        # but during inference we use the original image (no resize)
        self.seg_image_size = 1024
        
        # Performance optimization parameters
        self.use_fp16 = False                # Use FP16 mixed precision inference
        self.batch_size = 1                  # Batch inference size (only for supported models)
        
        # Segmentation model parameters
        self.seg_model = "imisnet"              # Segmentation model type ("sam", "medsam", "imisnet")
        self.undo_radius = 3                # Click undo radius
        self.use_previous_mask = True       # Use previous mask as input
        
        # Grounding model parameters
        self.grounding_model = "gpt"  # grounding model type (options: "qwen", "gpt")
        self.use_mask_module = True         # Use mask module
        
        # Analysis and logging parameters
        self.save_masks_history = False     # Save per-round mask history (will enlarge JSON file)
        self.compute_per_round_iou = True   # Compute per-round IoU in real time (requires GT mask)
        
        # History parameters (only valid when using qwen)
        self.max_history_length = 5        # Maximum history length
        self.use_history = True             # Use history
        self.reset_history_per_image = False # Reset history per image
        
        # Output directory parameters
        self.output_dir = "./intermediate_results"  # Directory for intermediate results and inference records
        self.results_dir = "./results"              # Directory for final aggregated results
        
        # Grounding model path
        self.model = None                     # Grounding model path (if applicable)
        self.checkpoint = None              # Segmentation model checkpoint path (generic)
        # Unified SAM/MedSAM config/checkpoint
        self.seg_checkpoint = None
        self.seg_config = None
        

class SingleImageInference:
    def __init__(self, grounding_model, segmentation_model, args):
        """
        Single-image inference class
        Args:
            grounding_model: Model used to generate points/boxes
            segmentation_model: Model used for segmentation  
            args: Configuration parameters
        """
        self.grounding_model = grounding_model
        self.segmentation_model = segmentation_model
        self.args = args
        self.workspace = getattr(args, 'output_dir')  # Use configured output dir
        self.dataset_name = getattr(args, 'dataset_name', 'default')  # Get dataset name from args
        
        # Initialize if using a model that supports history
        grounding_model_type = getattr(args, 'grounding_model', '')
        if 'qwen' in grounding_model_type or 'gpt' in grounding_model_type.lower():
            # Check for start_new_session method (confirm history support)
            if hasattr(self.grounding_model, 'start_new_session'):
                model_name = 'GPT' if 'gpt' in grounding_model_type.lower() else 'Qwen'
                print(f"Detected {model_name} model, enabling conversation history")
                self.use_history = True
            else:
                print(f"Warning: grounding_model set to {grounding_model_type} but model does not support history")
                self.use_history = False
        else:
            self.use_history = False
        
    def forward_single_image(self, img_path, target_description, max_clicks=5, 
                           visualize=True, gt_mask=None):
        """
        Perform multi-round interactive segmentation for a single image
        
        Args:
            img_path (str): Image path
            target_description (str): Target description
            max_clicks (int): Maximum number of clicks
            visualize (bool): Whether to save visualizations
            gt_mask (np.ndarray): Ground truth mask for real-time IoU computation
            
        Returns:
            final_mask: Final segmentation result
            inference_record: Inference process record
        """
        print(f"Start processing image: {img_path}")
        print(f"Target description: {target_description}")
        
        # Manage session if using history-aware model
        session_id = None
        if self.use_history:
            # Create session ID based on image path and description
            img_name = os.path.basename(img_path).split('.')[0]
            session_id = f"{img_name}_{hash(target_description) % 10000}"
            
            if getattr(self.args, 'reset_history_per_image', True):
                # Start a new session for each image
                session_id = self.grounding_model.start_new_session(session_id)
                print(f"Created new conversation session for image: {session_id}")
            else:
                # Switch to existing session or create new one
                self.grounding_model.switch_session(session_id)
                summary = self.grounding_model.get_session_summary()
                print(f"Using session {session_id}, current history length: {summary['current_history_length']} rounds")
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Cannot read image: {img_path}")
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Save original size
        original_height, original_width = image.shape[:2]
        
        # Record current image name for internal use
        self.args.current_image_name = os.path.basename(img_path).split('.')[0]

        # Resize for grounding: only apply to grounding model (in memory),
        # segmentation model uses original image (no resize) to keep segmentation resolution at args.seg_image_size (default 1024)
        grounding_resized_img_path = None
        orig_img_path = img_path
        grounding_img_input = orig_img_path
        if hasattr(self.args, 'grounding_resize') and self.args.grounding_resize is not None:
            target_size = self.args.grounding_resize
            print(f"Resize image for grounding model from {original_width}x{original_height} to {target_size}x{target_size}")

            image_pil = Image.fromarray(image_rgb)
            image_pil.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)

            resized_image = Image.new('RGB', (target_size, target_size), (255, 255, 255))
            offset = ((target_size - image_pil.size[0]) // 2, (target_size - image_pil.size[1]) // 2)
            resized_image.paste(image_pil, offset)

            resized_rgb = np.array(resized_image)
            resized_bgr = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)

            img_name = os.path.basename(img_path).split('.')[0]
            grounding_resized_img_path = f"{img_name}.png"
            grounding_img_input = resized_image
            print(f"Created in-memory grounding resize image: {grounding_resized_img_path}")
        
        height, width = image.shape[:2]
        
        # Initialize record
        inference_record = {
            "img_path": img_path,
            "target_description": target_description,
            "height": height,
            "width": width,
            "clicks_history": [],
            "model_outputs": [],
            "final_mask": None
        }
        
        # Add per_round_metrics if computing per-round IoU
        if getattr(self.args, 'compute_per_round_iou', False) and gt_mask is not None:
            inference_record["per_round_metrics"] = []
            # Preprocess GT mask
            if isinstance(gt_mask, Image.Image):
                gt_mask_np = np.array(gt_mask)
            else:
                gt_mask_np = gt_mask
            if len(gt_mask_np.shape) == 3:
                gt_mask_np = gt_mask_np[:, :, 0]
            gt_mask_binary = (gt_mask_np > 0).astype(np.uint8)
            print(f"Enabled real-time per-round IoU computation")
        else:
            gt_mask_binary = None
        
        # Prepare input data structure
        inputs = {
            # Image path passed to grounding (may be a temporary resize to grounding_resize)
            "img_path": orig_img_path,
            "caption": [target_description],
            "height": height,
            "width": width,
            "pred_list": []
        }
        
        # Initialize segmentation model (use original image, not grounding resize)
        simple_click_image = self.segmentation_model.image_process(img_path=orig_img_path)
        
        with torch.no_grad():
            self.segmentation_model.set_input_image(simple_click_image)
            
            clicker = Clicker(dataset_name=self.dataset_name, 
                            output_dir=getattr(self.args, 'output_dir'))
            previous_mask = None
            pred_logits = None
            last_ref_box_str = None
            current_box = None  # Store current box (from model-predicted bbox)
            retry_count = 0  # Retry counter
            max_retries = 1  # Max 1 retry
            actual_click_count = 0  # Actual valid click count
            
            print(f"Start multi-round interactive segmentation, max rounds: {max_clicks}")
            
            click_id = 0
            while actual_click_count < max_clicks:
                print(f"\n--- Round {actual_click_count + 1} (Attempt {click_id + 1}) ---")
                
                # Model predicts from the first round
                # If model outputs bbox, it will be saved to current_box for later use
                
                # Build prompt
                if self.use_history:
                    # Build prompt with history support
                    use_history = getattr(self.args, 'use_history', True)
                    reset_history = (actual_click_count == 0 and getattr(self.args, 'reset_history_per_image', True))
                    
                    prompt, conv = self.grounding_model.build_prompt(
                        inputs, 
                        last_ref_box_str, 
                        use_history=use_history,
                        reset_history=reset_history
                    )
                    print(f"Prompt built with history, history length: {len(self.grounding_model.conversation_history) // 2} rounds")
                else:
                    # Build regular prompt
                    prompt, conv = self.grounding_model.build_prompt(inputs, last_ref_box_str)
                
                # Generate model response
                mask_for_prompt = previous_mask if previous_mask is not None else np.zeros((height, width), dtype=np.uint8)
                
                if self.use_history:
                    # Generate response with history support (pass grounding resize image)
                    outputs = self.grounding_model.generate_response(
                        prompt, grounding_img_input, mask_for_prompt, conv, save_history=True, round_num=actual_click_count
                    )
                else:
                    outputs = self.grounding_model.generate_response(
                        prompt, grounding_img_input, mask_for_prompt, conv, round_num=actual_click_count
                    )
                
                if last_ref_box_str is not None:
                    outputs = last_ref_box_str + outputs
                    
                print(f"Model output: {outputs}")
                inference_record["model_outputs"].append(outputs)
                
                # Box chaining - XML tag format support removed
                # New format uses coordinates directly, no chaining needed
                last_ref_box_str = None

                # Parse output with process_response
                is_positive, points, should_stop = self.grounding_model.process_response(outputs)
                
                # Check if early stop is requested
                if should_stop:
                    if actual_click_count == 0 and retry_count < max_retries:
                        print("⚠️ Model requested stop in first round, retrying once...")
                        retry_count += 1
                        click_id += 1
                        continue
                    print("✅ Model requested stop, ending interaction early")
                    # Record stop operation
                    click_info = {"operation": "stop"}
                    self._update_record(inference_record, actual_click_count, current_box, 
                                    previous_mask, click_info, outputs, gt_mask_binary)
                    break
                
                # Handle bbox output
                if is_positive == 'bbox' and points is not None and len(points) == 4:
                    # Model output bbox
                    retry_count = 0
                    
                    # Convert bbox to absolute pixel coordinates
                    bbox_abs = [
                        int(round(points[0] * width)),   # x1
                        int(round(points[1] * height)),  # y1
                        int(round(points[2] * width)),   # x2
                        int(round(points[3] * height))   # y2
                    ]
                    print(f"Generated bbox - normalized: [{points[0]:.3f}, {points[1]:.3f}, {points[2]:.3f}, {points[3]:.3f}]")
                    print(f"Generated bbox - absolute: [{bbox_abs[0]}, {bbox_abs[1]}, {bbox_abs[2]}, {bbox_abs[3]}]")
                    
                    # Save bbox for later rounds (box first, then points)
                    current_box = bbox_abs
                    print(f"✓ Saved bbox for later rounds")
                    
                    # Segment with bbox directly, no points
                    pred_mask = self.segmentation_model.get_prediction(
                        clicker=None, box=bbox_abs, mask=pred_logits
                    )
                    
                    if isinstance(pred_mask, tuple):
                        pred_mask, pred_logits = pred_mask
                    
                    # Update mask
                    previous_mask = torch.from_numpy(pred_mask).to(torch.uint8).unsqueeze(0)
                    
                    # Record this round's result
                    click_info = {"operation": "add_bbox", "bbox": bbox_abs}  # Record operation type and bbox
                    self._update_record(inference_record, actual_click_count, bbox_abs, 
                                    previous_mask, click_info, outputs, gt_mask_binary)
                    
                    # Successful processing, increment counters
                    actual_click_count += 1
                    click_id += 1
                
                # Handle point output
                elif is_positive is not None and points is not None:
                    # Successfully parsed tool output, reset retry counter
                    retry_count = 0
                    
                    # Model output a point
                    # Convert point coordinates: points[0] is x, points[1] is y
                    abs_x = round(points[0] * width)
                    abs_y = round(points[1] * height)
                    abs_points = (abs_y, abs_x)  # Click class expects (y, x)
                    print(f"Generated click point - normalized: ({points[0]:.3f}, {points[1]:.3f})")
                    print(f"Generated click point - absolute: x={abs_x}, y={abs_y}, label: {is_positive}")
                    
                    # Create click and add to history
                    click = Click(is_positive=is_positive, coords=abs_points)
                    clicker.add_click(click, getattr(self.args, 'undo_radius', 3))
                    
                    # Box strategy:
                    box_to_use = current_box if pred_logits is None else None
                    
                    if pred_logits is None and current_box is not None:
                        print(f"✓ [First round] Segment with box + points, box: {current_box}, points: {len(clicker.clicks_list)}")
                    elif pred_logits is not None:
                        print(f"✓ [Refinement] Use points + previous mask, no box, points: {len(clicker.clicks_list)}")
                    else:
                        print(f"✓ [First round] Segment with points only, points: {len(clicker.clicks_list)}")
                    
                    pred_mask = self.segmentation_model.get_prediction(
                        clicker, box=box_to_use, mask=pred_logits
                    )
                    
                    if isinstance(pred_mask, tuple):
                        pred_mask, pred_logits = pred_mask
                    
                    # Update mask
                    previous_mask = torch.from_numpy(pred_mask).to(torch.uint8).unsqueeze(0)
                    
                    # Record this round's result
                    click_info = {"operation": "add_point", "is_positive": is_positive, "coords": abs_points}
                    self._update_record(inference_record, actual_click_count, current_box, 
                                    previous_mask, click_info, outputs, gt_mask_binary)
                    
                    actual_click_count += 1
                    click_id += 1
                
                else:
                    # Model output neither points nor stop request
                    retry_count += 1
                    print(f"⚠️ Warning: Invalid model output (no points and no stop): {outputs}")
                    print(f"   Current retries: {retry_count}/{max_retries}")
                    
                    if retry_count > max_retries:
                        print(f"❌ Reached max retries ({max_retries}), stopping inference")
                        break
                    else:
                        print(f"🔄 Retrying with the same prompt...")
                        # Retry: do not increment actual_click_count, only click_id
                        click_id += 1
                        continue
                
                # Visualize points and/or box for each round (including box-only rounds)
                if visualize and (len(clicker) > 0 or current_box is not None):
                    try:
                        point_list = [click.coords for click in clicker.clicks_list]
                        label_list = [click.is_positive for click in clicker.clicks_list]
                        self._save_pointlist_visualization(
                            image_rgb,
                            previous_mask,
                            point_list,
                            label_list,
                            img_path,
                            actual_click_count - 1,
                            outputs,
                            current_box,
                        )
                    except Exception as e:
                        print(f"Failed to save point list visualization: {e}")
            
            # End of loop summary
            print(f"\nInference loop finished:")
            print(f"  - Actual valid rounds: {actual_click_count}")
            print(f"  - Total attempts: {click_id}")
            print(f"  - Reached max rounds: {'Yes' if actual_click_count >= max_clicks else 'No'}")
            
            # Add history info
            if self.use_history and session_id:
                summary = self.grounding_model.get_session_summary()
                inference_record["session_info"] = {
                    "session_id": session_id,
                    "total_history_turns": summary["current_history_length"],
                    "max_history_length": summary["max_history_length"]
                }
                print(f"Session {session_id} completed, saved {summary['current_history_length']} history rounds")
            
            print(f"\nInference complete, {len(inference_record['clicks_history'])} rounds of interaction")
            
            return previous_mask, inference_record
    
    def _update_record(self, record, click_id, box, mask, click_info, outputs, gt_mask_binary=None):
        """Update inference record"""
        # Save per-round mask based on config
        if getattr(self.args, 'save_masks_history', False) and mask is not None:
            if isinstance(mask, torch.Tensor):
                mask_np = mask.squeeze(0).cpu().numpy()
            else:
                mask_np = mask
            
            # Add mask to history
            if "masks_history" not in record:
                record["masks_history"] = []
            record["masks_history"].append(mask_np.tolist())
        
        # Compute per-round IoU in real time (if enabled and GT mask provided)
        if getattr(self.args, 'compute_per_round_iou', False) and gt_mask_binary is not None and mask is not None:
            if isinstance(mask, torch.Tensor):
                mask_np = mask.squeeze(0).cpu().numpy()
            else:
                mask_np = mask
            
            # Resize to match GT
            if mask_np.shape != gt_mask_binary.shape:
                from PIL import Image
                mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
                mask_resized = mask_pil.resize((gt_mask_binary.shape[1], gt_mask_binary.shape[0]), Image.LANCZOS)
                mask_np = (np.array(mask_resized) > 128).astype(np.uint8)
            
            # Compute IoU and Dice
            dice, iou = get_metrics(mask_np, gt_mask_binary)
            
            # Add to per_round_metrics
            if "per_round_metrics" not in record:
                record["per_round_metrics"] = []
            # Get operation type from click_info
            operation_type = click_info.get('operation', 'unknown') if isinstance(click_info, dict) else 'unknown'
            record["per_round_metrics"].append({
                "round": click_id,
                "operation": operation_type,
                "iou": float(iou),
                "dice": float(dice)
            })
            print(f"  Round {click_id} ({operation_type}) - IoU: {iou:.4f}, Dice: {dice:.4f}")
        
        record["clicks_history"].append({
            "click_id": click_id,
            "click_info": click_info,
            "used_box": box,
            "outputs": outputs
        })
    
    def _save_visualization(self, image, mask, point, img_path, click_id, stage, 
                          caption, is_positive=True):
        """Save visualization - disabled, no longer saves individual step images"""
        # No longer save individual step images or txt files
        pass
    
    def _save_box_only_visualization(self, image, mask, box, img_path, click_id, caption):
        """Save box-only visualization - disabled"""
        # No longer save box-only images or txt files
        pass
    
    def _save_pointlist_visualization(self, image, mask, point_list, label_list, 
                                    img_path, click_id, outputs, box):
        """Save point list visualization"""
        try:
            sample_output_dir = getattr(self.args, 'sample_output_dir', None)
            if sample_output_dir:
                vis_dir = sample_output_dir
            else:
                img_name = os.path.basename(img_path).split('.')[0]
                vis_dir = os.path.join(self.workspace, img_name)
            os.makedirs(vis_dir, exist_ok=True)
            save_path = os.path.join(vis_dir, f"step_{click_id:02d}.jpg")
            
            visualize_mask_and_pointlist(
                image, mask, point_list, save_path, label_list, outputs, box
            )
        except Exception as e:
            print(f"Failed to save point list visualization: {e}")
    
    def save_inference_record(self, record, save_path=None):
        """Save inference record (simplified, without mask data)"""
        if save_path is None:
            sample_output_dir = getattr(self.args, 'sample_output_dir', None)
            if sample_output_dir:
                save_path = os.path.join(sample_output_dir, "inference_record.json")
            else:
                img_name = os.path.basename(record["img_path"]).split('.')[0]
                save_path = os.path.join(self.workspace, img_name, "inference_record.json")
        
        # Create simplified record with only key information
        record_simplified = {
            "img_path": record["img_path"],
            "target_description": record["target_description"],
            "height": record["height"],
            "width": record["width"],
            "num_clicks": len(record["clicks_history"]),
            "clicks_history": record["clicks_history"],
            "model_outputs": record["model_outputs"],
            "per_round_metrics": record.get("per_round_metrics", []),
            "session_info": record.get("session_info", {})
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(record_simplified, f, indent=2, ensure_ascii=False)
        
        # print(f"Inference record saved to: {save_path}")
        return save_path


def run_single_image_inference(img_path, target_description, grounding_model, 
                              segmentation_model, args, max_clicks=5):
    """
    Convenience function to run single-image inference
    
    Args:
        img_path (str): Image path
        target_description (str): Target description
        grounding_model: grounding model instance
        segmentation_model: segmentation model instance
        args: Configuration parameters
        max_clicks (int): Maximum number of clicks
    Returns:
        final_mask: Final segmentation result
        record_path: Inference record save path
    """
    # Create inferencer
    inferencer = SingleImageInference(grounding_model, segmentation_model, args)
    
    # Run inference
    final_mask, record = inferencer.forward_single_image(
        img_path=img_path,
        target_description=target_description,
        max_clicks=max_clicks,
        visualize=True
    )
    
    # Save record
    record_path = inferencer.save_inference_record(record)
    
    return final_mask, record_path
