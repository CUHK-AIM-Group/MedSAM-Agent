# Copyright 2025
# MedSAM-Agent reward function based on format validation and IoU

import logging
import re
import os
import numpy as np
from PIL import Image
from typing import Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Global counter for saving masks
_mask_save_counter = 0
_mask_save_dir = os.path.join("outputs", datetime.now().strftime("%Y-%m-%d"), "debug_masks")


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute IoU between predicted mask and ground truth mask.
    
    Args:
        pred_mask: binary mask array (H, W) with values 0 or 255
        gt_mask: binary mask array (H, W) with values 0 or 255
        
    Returns:
        IoU score in [0, 1]
    """
    # Binarize masks (handle both 0/1 and 0/255 conventions)
    pred_binary = (pred_mask > 127).astype(np.uint8)
    gt_binary = (gt_mask > 127).astype(np.uint8)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    if union == 0:
        # Both masks are empty
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    return float(iou)


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute Dice coefficient between predicted mask and ground truth mask.
    
    Dice coefficient is more sensitive to small objects compared to IoU,
    making it particularly suitable for medical image segmentation.
    
    Args:
        pred_mask: binary mask array (H, W) with values 0 or 255
        gt_mask: binary mask array (H, W) with values 0 or 255
        
    Returns:
        Dice score in [0, 1]
    """
    # Binarize masks (handle both 0/1 and 0/255 conventions)
    pred_binary = (pred_mask > 127).astype(np.uint8)
    gt_binary = (gt_mask > 127).astype(np.uint8)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    pred_sum = pred_binary.sum()
    gt_sum = gt_binary.sum()
    
    if pred_sum + gt_sum == 0:
        # Both masks are empty
        return 1.0 if intersection == 0 else 0.0
    
    dice = (2.0 * intersection) / (pred_sum + gt_sum)
    return float(dice)


def save_masks_for_debug(pred_mask_np: np.ndarray, gt_mask_np: np.ndarray, 
                         iou_score: float, index: Optional[int] = None,
                         step: Optional[int] = None, save_dir: Optional[str] = None) -> None:
    """Save predicted mask, ground truth mask, and overlay for visualization.
    
    Args:
        pred_mask_np: Predicted mask as numpy array
        gt_mask_np: Ground truth mask as numpy array
        iou_score: IoU score for this pair
        index: Sample index in the batch (for filename)
        step: Training step (for filename)
        save_dir: Directory to save masks (if None, use global default)
    """
    global _mask_save_counter
    
    try:
        # Determine save directory
        if save_dir is None:
            save_dir = _mask_save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate filename with counter, step, and index
        _mask_save_counter += 1
        timestamp = datetime.now().strftime("%H%M%S")
        
        # Build filename components
        filename_parts = [f"sample_{_mask_save_counter:05d}"]
        if step is not None:
            filename_parts.append(f"step_{step}")
        if index is not None:
            filename_parts.append(f"idx_{index}")
        filename_parts.append(f"iou_{iou_score:.4f}")
        filename_parts.append(timestamp)
        
        filename_base = "_".join(filename_parts)
        
        # Save predicted mask
        pred_path = os.path.join(save_dir, f"{filename_base}_pred.png")
        Image.fromarray(pred_mask_np).save(pred_path)
        
        # Save ground truth mask
        gt_path = os.path.join(save_dir, f"{filename_base}_gt.png")
        Image.fromarray(gt_mask_np).save(gt_path)
        
        # Create overlay visualization (RGB image)
        # Green: True Positive (both pred and gt)
        # Red: False Positive (pred only)
        # Blue: False Negative (gt only)
        # Black: True Negative (neither)
        overlay = np.zeros((*pred_mask_np.shape, 3), dtype=np.uint8)
        
        pred_binary = (pred_mask_np > 127).astype(bool)
        gt_binary = (gt_mask_np > 127).astype(bool)
        
        # True Positive - Green
        overlay[pred_binary & gt_binary] = [0, 255, 0]
        # False Positive - Red
        overlay[pred_binary & ~gt_binary] = [255, 0, 0]
        # False Negative - Blue
        overlay[~pred_binary & gt_binary] = [0, 0, 255]
        
        overlay_path = os.path.join(save_dir, f"{filename_base}_overlay.png")
        Image.fromarray(overlay).save(overlay_path)
        
        # Save a side-by-side comparison
        h, w = pred_mask_np.shape
        comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
        
        # Convert grayscale to RGB for visualization
        pred_rgb = np.stack([pred_mask_np] * 3, axis=-1)
        gt_rgb = np.stack([gt_mask_np] * 3, axis=-1)
        
        comparison[:, :w] = gt_rgb
        comparison[:, w:2*w] = pred_rgb
        comparison[:, 2*w:] = overlay
        
        comparison_path = os.path.join(save_dir, f"{filename_base}_comparison.png")
        Image.fromarray(comparison).save(comparison_path)
        
        logger.debug(f"Saved debug masks to {save_dir}/{filename_base}_*.png")
        
    except Exception as e:
        logger.warning(f"Failed to save debug masks: {e}")


def compute_score(data_source: str, solution_str: str, ground_truth: Any, extra_info: Optional[dict] = None, **kwargs) -> float:
    """
    Compute reward score for MedSAM-Agent based on format validation and segmentation quality.
    
    Returns a weighted combination of:
    - Format reward (0.2 weight): Graded based on tool usage correctness
    - Quality reward (0.8 weight): Combined metric of IoU and efficiency
      * Quality = IoU + Progression penalties, clamped to [0, 1]
      * Higher IoU increases quality
      * Overshoot and excessive turns decrease quality
    
    Final score range: [0.0, 1.0]
    
    Format reward rules:
    - Both conditions met (interaction + stop): 1.0
    - Only one condition met: 0.5 (penalty of -0.5)
    - No conditions met: 0.0
    
    Progression penalty rules (encourages efficiency):
    - Start from 0.0 (no penalty)
    - Overshoot penalty (not stopping at optimal point):
      * If max IoU is not at the final turn
      * Penalty = (max_IoU - final_IoU) × 1.0
    - Turn count penalty (encourage fewer turns):
      * 0.01 per turn
      * e.g., 2 turns: 0.02, 5 turns: 0.05
    - Quality = max(0, min(1, IoU - total_penalty))
    
    Args:
        data_source: Source identifier (not used currently)
        solution_str: Model's generated solution text
        ground_truth: Ground truth mask (PIL Image or numpy array)
        extra_info: Dictionary containing:
            - 'pred_mask': Predicted mask(s). Can be:
                * List of masks (multi-turn): [mask1, mask2, ...] where each is PIL Image or numpy array
                * Single mask (backward compatibility): PIL Image or numpy array
            - 'stopped': Whether stop_action was called (bool)
            - 'num_turns': Number of refinement turns completed (int)
            - 'conversation_history': Full conversation history as list of message dicts
                [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]
            - 'multi_modal_data': Dict containing image data, e.g., {"image": [PIL.Image, ...]}
            - 'question': Optional question text for logging
            - 'index': Sample index in batch (for debug filenames)
            - 'step': Training step (for debug filenames)
            - 'save_masks': Whether to save masks for debugging (default: True)
            - 'mask_save_dir': Custom directory to save masks (optional)
            - 'ground_truth_mask': Alternative source for ground truth mask
            - Other metadata from dataset
        **kwargs: Additional keyword arguments (e.g., reward_router_address, reward_model_tokenizer)
                 that may be passed by the framework but are not used in this implementation
    
    Returns:
        Dictionary containing:
        - 'score': Final weighted score in [0, 1] (format: 20%, quality: 80%)
        - 'iou': IoU value (final turn)
        - 'format_reward': Format reward (1.0, 0.5, or 0.0)
        - 'progression_penalty': Total penalty value (range: [0, +∞))
        - 'iou_per_turn': List of IoU scores for each turn [turn1_iou, turn2_iou, ...]
    """
    # Ensure logger level allows INFO messages for debugging
    if logger.level > logging.INFO or logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    
    extra_info = extra_info or {}
    
    # Extract debug parameters
    sample_index = extra_info.get("index")
    training_step = extra_info.get("step")
    should_save_masks = extra_info.get("save_masks", True)  # Default: save masks
    custom_save_dir = extra_info.get("mask_save_dir")
    ground_truth = extra_info.get("ground_truth_mask", ground_truth)
    num_turns = extra_info.get("num_turns")
    
    # ========== Hyperparameter Configuration ==========
    # Flexible reward function hyperparameters for easy tuning
    
    # === Main reward weights ===
    format_weight = 0.2  # Weight for format validation (0.0 - 1.0)
    quality_weight = 0.8  # Weight for quality metrics (0.0 - 1.0)
    # Note: format_weight + quality_weight should = 1.0
    
    # === Metric combination switches ===
    enable_dice = True  # Enable Dice coefficient in addition to IoU
    enable_improvement_reward = True  # Enable progressive improvement bonus

    # === Penalty switches ===
    enable_turn_cost = True  # Enable per-turn efficiency penalty
    enable_overshoot_penalty = True  # Enable penalty for not stopping at max IoU
    
    # === Metric combination weights (only used if enabled) ===
    iou_weight = 0.5  # Weight for IoU in combined metric (0.0 - 1.0)
    dice_weight = 0.5  # Weight for Dice in combined metric (0.0 - 1.0)
    # Note: iou_weight + dice_weight should = 1.0 when both enabled
    
    # === Progressive improvement reward ===
    improvement_bonus_scale = 0.1  # Multiplier for IoU improvements (0.0 - 1.0)
    # Each IoU increase is rewarded: improvement_bonus += delta_iou * improvement_bonus_scale
    
    # === Penalty scales ===
    turn_penalty_scale = 0.01  # Penalty per turn (e.g., 0.01 → 3 turns = 0.03 penalty)
    overshoot_penalty_scale = 1.0  # Multiplier for IoU gap penalty (1.0 = 1:1 mapping)
    # Overshoot penalty = (max_iou - final_iou) * overshoot_penalty_scale

    pred_mask_input = extra_info.get("pred_mask")
    logger.debug(f"[INPUT] pred_mask: list with {len(pred_mask_input)} masks")

    # Initialize tracking variables
    format_reward = 0.0
    iou_reward = 0.0
    dice_reward = 0.0
    iou_per_turn = []  # Track IoU for each turn
    dice_per_turn = []  # Track Dice for each turn
    progression_penalty = 0.0  # Total penalty for inefficient progression
    improvement_bonus = 0.0  # Bonus for progressive improvements
    
    # ===== 1. Format validation =====
    # Format reward: 1.0 if (1) last tool call is stop_action AND (2) at least one add_bbox/add_point
    # Find all tool_call blocks
    tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
    tool_calls = re.findall(tool_call_pattern, solution_str, re.DOTALL)
    
    logger.debug(f"[FORMAT] Found {len(tool_calls)} tool calls")
    
    if tool_calls:
        import json
        interaction_tools = {"add_bbox", "add_point"}
        
        has_interaction = False  # Has at least one add_bbox or add_point
        ends_with_stop = False   # Last call is stop_action
        
        # Check for at least one interaction tool
        for tool_call_str in tool_calls:
            try:
                tool_data = json.loads(tool_call_str.strip())
                tool_name = tool_data.get("name", "")
                logger.debug(f"[FORMAT] Checking tool: {tool_name}")
                if tool_name in interaction_tools:
                    has_interaction = True
                    logger.debug(f"[FORMAT] Found interaction tool: {tool_name}")
                    break
            except:
                # Fallback to string matching
                if "add_bbox" in tool_call_str or "add_point" in tool_call_str:
                    has_interaction = True
                    logger.debug(f"[FORMAT] Found interaction tool (fallback)")
                    break
        
        # Check if last tool call is stop_action
        last_tool_call = tool_calls[-1].strip()
        try:
            tool_data = json.loads(last_tool_call)
            if isinstance(tool_data, dict) and tool_data.get("name") == "stop_action":
                ends_with_stop = True
                logger.debug(f"[FORMAT] Last tool is stop_action")
        except:
            # Fallback to string matching
            if "stop_action" in last_tool_call:
                ends_with_stop = True
                logger.debug(f"[FORMAT] Last tool is stop_action (fallback)")
        
        # Calculate format reward with partial credit
        if has_interaction and ends_with_stop:
            format_reward = 1.0
            logger.debug("[FORMAT] Final format_reward: 1.0 (has add_bbox/add_point AND ends with stop_action)")
        elif has_interaction or ends_with_stop:
            format_reward = 0.5
            if has_interaction and not ends_with_stop:
                logger.debug("[FORMAT] Final format_reward: 0.5 (has add_bbox/add_point but missing stop_action)")
            else:
                logger.debug("[FORMAT] Final format_reward: 0.5 (has stop_action but missing add_bbox/add_point interaction)")
        else:
            format_reward = 0.0
            logger.debug("[FORMAT] Final format_reward: 0.0 (no interactions and no stop_action)")
    else:
        format_reward = 0.0
        logger.debug("[FORMAT] Final format_reward: 0.0 (no tool calls found)")
    
    # ===== 2. IoU calculation =====
    
    logger.debug("[IOU] Starting IoU calculation")
    
    # Get predicted masks (can be a list of masks or a single mask for backward compatibility)
    # Use the normalized pred_mask_input to ensure object arrays are handled correctly
    pred_mask = pred_mask_input
    
    # Determine if pred_mask is a list of masks (multi-turn) or single mask
    if isinstance(pred_mask, list) and len(pred_mask) > 0:
        pred_masks = pred_mask  # List of masks from all turns
        logger.debug(f"[IOU] Found {len(pred_masks)} masks (multi-turn)")
    elif pred_mask is not None:
        pred_masks = [pred_mask]  # Single mask, wrap in list
        logger.debug(f"[IOU] Found single mask (backward compatibility)")
    else:
        pred_masks = []
        logger.debug(f"[IOU] No pred_mask found")

    # Safety: cap unreasonable history lengths (e.g., batched object arrays leaking through)
    if len(pred_masks) > 0:
        # Prefer num_turns hint if provided; otherwise default cap 32
        max_allowed_turns = None
        if isinstance(num_turns, int) and num_turns > 0:
            max_allowed_turns = num_turns
        default_cap = 32
        cap = max_allowed_turns or default_cap
        if len(pred_masks) > cap:
            logger.warning(
                f"[IOU] Pred mask history length {len(pred_masks)} exceeds cap {cap}, truncating to first {cap} entries"
            )
            pred_masks = pred_masks[:cap]
    
    if ground_truth is None:
        logger.warning("No ground truth mask provided. IoU reward = 0")
        iou_reward = 0.0
    elif len(pred_masks) > 0:
        # Calculate IoU for each turn
        logger.debug(f"Computing IoU for {len(pred_masks)} turns")
        
        try:
            # Convert ground truth once
            if isinstance(ground_truth, Image.Image):
                gt_mask_np = np.array(ground_truth)
            else:
                gt_mask_np = np.array(ground_truth)
            
            # Handle multi-channel ground truth
            if len(gt_mask_np.shape) == 3:
                gt_mask_np = gt_mask_np[:, :, 0]
            
            # Ensure uint8
            if gt_mask_np.dtype != np.uint8:
                gt_mask_np = gt_mask_np.astype(np.uint8)
            
            # Process each mask in history
            for turn_idx, turn_mask in enumerate(pred_masks, 1):
                try:
                    # Convert mask to numpy
                    if isinstance(turn_mask, Image.Image):
                        turn_mask_np = np.array(turn_mask)
                    else:
                        turn_mask_np = np.array(turn_mask)
                    
                    # Handle multi-channel
                    if len(turn_mask_np.shape) == 3:
                        turn_mask_np = turn_mask_np[:, :, 0]
                    
                    # Ensure uint8
                    if turn_mask_np.dtype != np.uint8:
                        turn_mask_np = turn_mask_np.astype(np.uint8)
                    
                    # Resize if needed
                    if turn_mask_np.shape != gt_mask_np.shape:
                        turn_pil = Image.fromarray(turn_mask_np, mode='L').resize(
                            (gt_mask_np.shape[1], gt_mask_np.shape[0]), 
                            Image.NEAREST
                        )
                        turn_mask_np = np.array(turn_pil)
                    
                    # Compute IoU for this turn
                    turn_iou = compute_iou(turn_mask_np, gt_mask_np)
                    iou_per_turn.append(turn_iou)
                    
                    # Compute Dice for this turn (if enabled)
                    if enable_dice:
                        turn_dice = compute_dice(turn_mask_np, gt_mask_np)
                        dice_per_turn.append(turn_dice)
                        logger.debug(f"[METRICS] Turn {turn_idx} - IoU: {turn_iou:.4f}, Dice: {turn_dice:.4f}")
                    else:
                        logger.debug(f"[IOU] Turn {turn_idx} IoU: {turn_iou:.4f}")
                    
                except Exception as e:
                    logger.error(f"[METRICS] Error computing metrics for turn {turn_idx}: {e}")
                    iou_per_turn.append(0.0)
                    if enable_dice:
                        dice_per_turn.append(0.0)
            
            # Final IoU reward is the last turn's IoU
            iou_reward = iou_per_turn[-1] if iou_per_turn else 0.0
            
            # Final Dice reward is the last turn's Dice (if enabled)
            if enable_dice and dice_per_turn:
                dice_reward = dice_per_turn[-1]
                logger.debug(f"[METRICS] Final scores - IoU: {iou_reward:.4f}, Dice: {dice_reward:.4f}")
            else:
                dice_reward = 0.0
                logger.debug(f"[IOU] Final IoU score (last turn): {iou_reward:.4f}")
            
            # Calculate progression penalty
            # Philosophy: Encourage stopping at the right time with minimal turns
            # - Penalize not stopping at max IoU (overshoot penalty)
            # - Penalize using too many turns (efficiency penalty)
            logger.debug("[PROGRESSION] Starting progression penalty calculation")
            
            if len(iou_per_turn) > 1:
                # Find the turn with maximum IoU
                max_iou = max(iou_per_turn)
                final_iou = iou_per_turn[-1]
                is_max_at_end = (abs(max_iou - final_iou) < 1e-6)  # Account for floating point precision
                
                logger.debug(f"[PROGRESSION] max_iou: {max_iou:.4f}, final_iou: {final_iou:.4f}, is_max_at_end: {is_max_at_end}")
                
                # === Progression penalty calculation ===
                # Directly compute penalty values (positive numbers)
                
                # Penalty 1: Overshoot penalty (not stopping at maximum IoU)
                overshoot_penalty = 0.0
                if not enable_overshoot_penalty:
                    logger.debug("[PROGRESSION] Overshoot penalty disabled via enable_overshoot_penalty flag")
                elif not is_max_at_end:
                    iou_gap = max_iou - final_iou
                    # Penalty proportional to the IoU gap with configurable scale
                    # e.g., gap=0.1, scale=1.0 → penalty=0.1
                    overshoot_penalty = iou_gap * overshoot_penalty_scale
                    
                    max_iou_turn = iou_per_turn.index(max_iou) + 1  # 1-indexed
                    
                    logger.debug(
                        f"[PROGRESSION] Max IoU {max_iou:.4f} at turn {max_iou_turn}/{len(iou_per_turn)}, "
                        f"final IoU {final_iou:.4f}, gap: {iou_gap:.4f}. "
                        f"Overshoot penalty: {overshoot_penalty:.4f}"
                    )
                else:
                    logger.debug(f"[PROGRESSION] Max IoU at final turn (optimal stopping)")
                
                # Penalty 2: Turn count penalty (encourage fewer turns)
                # Can be disabled via enable_turn_cost flag
                num_turns = len(iou_per_turn)
                if not enable_turn_cost:
                    turn_penalty = 0.0
                else:
                    turn_penalty = num_turns * turn_penalty_scale
                # Total penalty
                progression_penalty = overshoot_penalty + turn_penalty
            else:
                # Single turn: no penalties
                progression_penalty = 0.0
            
            # ===== Calculate progressive improvement reward =====
            # Reward each step that improves IoU over the previous step
            if enable_improvement_reward and len(iou_per_turn) > 1:
                for i in range(1, len(iou_per_turn)):
                    delta = iou_per_turn[i] - iou_per_turn[i-1]
                    if delta > 0:
                        step_bonus = delta * improvement_bonus_scale
                        improvement_bonus += step_bonus
                        logger.debug(f"[IMPROVEMENT] Turn {i} → {i+1}: IoU +{delta:.4f}, bonus +{step_bonus:.4f}")
            else:
                improvement_bonus = 0.0
                if enable_improvement_reward:
                    logger.debug("[IMPROVEMENT] Single turn or no improvements, bonus = 0.0")
                else:
                    logger.debug("[IMPROVEMENT] Improvement reward disabled")
            
        except Exception as e:
            logger.error(f"Error computing per-turn IoU: {e}", exc_info=True)
            iou_reward = 0.0
    else:
        logger.warning("No predicted masks found in extra_info. IoU reward = 0")
        iou_reward = 0.0
    
    # ===== 3. Final weighted score =====
    # Combine metrics into a single quality score
    
    # Step 1: Combine IoU and Dice (if enabled) into a base metric
    if enable_dice and dice_reward > 0:
        # Weighted combination of IoU and Dice
        base_metric = iou_weight * iou_reward + dice_weight * dice_reward
        logger.debug(f"[QUALITY] Base metric: {base_metric:.4f} (IoU: {iou_reward:.4f} × {iou_weight}, Dice: {dice_reward:.4f} × {dice_weight})")
    else:
        # Use IoU only
        base_metric = iou_reward
        logger.debug(f"[QUALITY] Base metric: {base_metric:.4f} (IoU only)")
    
    # Step 2: Apply penalties and bonuses
    # Quality = base_metric - penalties + bonuses, clamped to [0, 1]
    combined_quality_reward = max(0.0, min(1.0, base_metric - progression_penalty + improvement_bonus))
    
    logger.debug(
        f"[QUALITY] Quality calculation: base={base_metric:.4f}, penalty={progression_penalty:.4f}, "
        f"bonus={improvement_bonus:.4f}, final={combined_quality_reward:.4f}"
    )
    
    # Step 3: Final weighted score: Format + Quality
    final_score = format_weight * format_reward + quality_weight * combined_quality_reward
    
    logger.info(
        f"[FINAL] Reward - Format: {format_reward:.2f} (×{format_weight}), "
        f"IoU: {iou_reward:.4f}, Dice: {dice_reward:.4f}, "
        f"Penalty: {progression_penalty:.4f}, Bonus: {improvement_bonus:.4f}, "
        f"Quality: {combined_quality_reward:.4f} (×{quality_weight}), Final: {final_score:.4f}"
    )

    # Return a dict so reward managers can collect extra info
    return {
        "score": float(final_score),
        "iou": float(iou_reward),
        "dice": float(dice_reward),
        "format_reward": float(format_reward),
        "progression_penalty": float(progression_penalty),
        "improvement_bonus": float(improvement_bonus),
        "combined_quality": float(combined_quality_reward),
        "iou_per_turn": [float(x) for x in iou_per_turn],
        "dice_per_turn": [float(x) for x in dice_per_turn] if dice_per_turn else []
    }
