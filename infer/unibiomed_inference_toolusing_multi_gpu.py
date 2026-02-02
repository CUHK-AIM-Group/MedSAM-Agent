"""
Multi-GPU multi-process inference script
Uses torch.multiprocessing for parallel inference, each process uses a single GPU independently
"""
import json
import os
import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from PIL import Image
from copy import deepcopy
import time
import argparse

# Import existing inference modules
from unibiomed_inference_toolusing import (
    InferenceArgs,
    SingleImageInference,
    get_metrics,
    get_mean,
    save_metrics_json
)


def setup_process_gpu(rank, n_gpus, processes_per_gpu):
    """
    Set GPU for each process
    Args:
        rank: Process index (0 to n_gpus*processes_per_gpu-1)
        n_gpus: Total GPU count
        processes_per_gpu: Processes per GPU
    """
    # Compute the GPU ID for this process
    gpu_id = rank // processes_per_gpu
    process_id_on_gpu = rank % processes_per_gpu
    
    # Set the current device
    torch.cuda.set_device(gpu_id)
    print(f"Process {rank} (GPU {gpu_id}, Process #{process_id_on_gpu} on this GPU) using cuda:{gpu_id}")


def worker_process(rank, n_gpus, processes_per_gpu, annotations_chunk, args, return_dict):
    """
    Inference task executed by each process
    Args:
        rank: Process index
        n_gpus: Total GPU count
        processes_per_gpu: Processes per GPU
        annotations_chunk: Data list assigned to this process
        args: Inference configuration
        return_dict: Shared dictionary for returning results
    """
    try:
        # Detect GPT mode
        is_gpt_mode = hasattr(args, 'gpt_model') or (isinstance(args.grounding_model, str) and 'gpt' in args.grounding_model.lower())
        
        # Set GPU only in non-GPT mode
        if not is_gpt_mode:
            setup_process_gpu(rank, n_gpus, processes_per_gpu)
            gpu_id = rank // processes_per_gpu
            process_id_on_gpu = rank % processes_per_gpu
            print(f"[Process {rank} @ GPU {gpu_id}] Process started, handling {len(annotations_chunk)} samples")
            print(f"[Process {rank} @ GPU {gpu_id}] Current CUDA device: {torch.cuda.current_device()}")
            print(f"[Process {rank} @ GPU {gpu_id}] Available GPU count: {torch.cuda.device_count()}")
        else:
            print(f"[Process {rank}] GPT mode process started, handling {len(annotations_chunk)} samples")
            gpu_id = None  # GPT mode does not use GPU
            process_id_on_gpu = 0
        
        # Deep copy args to create per-process config
        args_copy = deepcopy(args)
        # Inform model loader of the desired GPU index for this process (not needed in GPT mode)
        if not is_gpt_mode:
            args_copy.device = gpu_id
        
        # Load models on the current GPU
        from models.model_loader import load_model
        segmentation_model, grounding_model = load_model(args_copy)
        
        if is_gpt_mode:
            print(f"[Process {rank}] GPT model loaded")
        else:
            print(f"[Process {rank} @ GPU {gpu_id}] Model loaded")
        
        # Create inferencer
        inferencer = SingleImageInference(grounding_model, segmentation_model, args_copy)
        
        # Statistics
        processed_count = 0
        failed_count = 0
        local_results = {}
        local_metrics = {}  # Store metrics for each image
        
        def build_sample_output_dir(item, image_path, base_output_dir):
            img_base = os.path.splitext(os.path.basename(image_path))[0]
            sample_id = item.get('id')
            if sample_id is not None:
                folder_name = f"{img_base}_{sample_id}"
            else:
                mask_base = os.path.splitext(os.path.basename(item.get('mask_file', '')))[0]
                folder_name = f"{img_base}_{mask_base}" if mask_base else img_base
            return os.path.join(base_output_dir, 'samples', folder_name)

        # Process assigned data
        desc = f"Process {rank} (GPT)" if is_gpt_mode else f"Process {rank} @ GPU {gpu_id}"
        for i, each_item in enumerate(tqdm(annotations_chunk, 
                                           desc=desc, 
                                           position=rank)):
            try:
                # Build file paths
                image_file = os.path.join(args.val_folder, each_item['split'], 
                                        each_item['file_name'])
                mask_path = os.path.join(args.val_folder, 
                                       each_item['split'] + '_mask', 
                                       each_item['mask_file'])
                
                # Check file existence
                if not os.path.exists(image_file):
                    log_prefix = f"[Process {rank}]" if is_gpt_mode else f"[GPU {rank}]"
                    print(f"{log_prefix} Error: Image file not found: {image_file}")
                    failed_count += 1
                    continue
                    
                if not os.path.exists(mask_path):
                    log_prefix = f"[Process {rank}]" if is_gpt_mode else f"[GPU {rank}]"
                    print(f"{log_prefix} Error: Mask file not found: {mask_path}")
                    failed_count += 1
                    continue
                
                # Read image and mask
                image = Image.open(image_file).convert('RGB')
                mask_file = Image.open(mask_path).convert('L')
                
                target_description = each_item['sentences'][0]['sent']

                # Prepare a separate output directory for each sample
                if args_copy.save_intermediate:
                    sample_output_dir = build_sample_output_dir(each_item, image_file, args_copy.output_dir)
                    args_copy.sample_output_dir = sample_output_dir
                    os.makedirs(sample_output_dir, exist_ok=True)

                    # Save original image
                    try:
                        original_path = os.path.join(sample_output_dir, "original_image.png")
                        if not os.path.exists(original_path):
                            image.save(original_path)
                    except Exception as e:
                        log_prefix = f"[Process {rank}]" if is_gpt_mode else f"[GPU {rank}]"
                        print(f"{log_prefix} Warning: Failed to save original image: {e}")

                    # Save GT mask and visualization
                    try:
                        gt_mask_np = np.array(mask_file)
                        if len(gt_mask_np.shape) == 3:
                            gt_mask_np = gt_mask_np[:, :, 0]
                        gt_mask_binary = (gt_mask_np > 0).astype(np.uint8) * 255
                        gt_mask_path = os.path.join(sample_output_dir, "gt_mask.png")
                        if not os.path.exists(gt_mask_path):
                            Image.fromarray(gt_mask_binary).save(gt_mask_path)

                        image_rgb = cv2.imread(image_file)
                        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
                        from utils.visual_utils import overlay_mask
                        gt_vis = overlay_mask(image_rgb, gt_mask_np)
                        gt_vis_path = os.path.join(sample_output_dir, "gt_overlay.png")
                        if not os.path.exists(gt_vis_path):
                            cv2.imwrite(gt_vis_path, cv2.cvtColor(gt_vis, cv2.COLOR_RGB2BGR))
                    except Exception as e:
                        log_prefix = f"[Process {rank}]" if is_gpt_mode else f"[GPU {rank}]"
                        print(f"{log_prefix} Warning: Failed to save GT result: {e}")
                else:
                    args_copy.sample_output_dir = None
                
                # Run inference and pass GT mask for real-time IoU calculation
                final_mask, record_path = inferencer.forward_single_image(
                    img_path=image_file,
                    target_description=target_description,
                    max_clicks=args_copy.n_clicks,
                    visualize=args_copy.save_intermediate,
                    gt_mask=mask_file  # Pass GT mask for real-time IoU calculation
                ), None
                
                # Save inference record (if enabled)
                if final_mask[0] is not None:
                    inference_record = final_mask[1]  # Keep record reference
                    record_path = None
                    if args_copy.save_intermediate:
                        record_path = inferencer.save_inference_record(inference_record)
                    final_mask = final_mask[0]
                else:
                    log_prefix = f"[Process {rank}]" if is_gpt_mode else f"[GPU {rank}]"
                    print(f"{log_prefix} Error: Inference failed, no valid mask generated")
                    failed_count += 1
                    continue
                
                # Resize mask and save
                try:
                    if isinstance(final_mask, torch.Tensor):
                        final_mask_np = final_mask.squeeze(0).cpu().numpy()
                    else:
                        final_mask_np = final_mask
                    
                    orig_h, orig_w = mask_file.size[::-1]
                    pred_mask_pil = Image.fromarray((final_mask_np * 255).astype(np.uint8))
                    pred_mask_resized = pred_mask_pil.resize((orig_w, orig_h), Image.LANCZOS)
                    pred_mask_final = np.array(pred_mask_resized) > 128
                    
                    # Compute IoU and Dice for the current image immediately
                    try:
                        gt_mask_np = np.array(mask_file)
                        if len(gt_mask_np.shape) == 3:
                            gt_mask_np = gt_mask_np[:, :, 0]
                        gt_mask_binary = (gt_mask_np > 0).astype(np.uint8)
                        pred_mask_binary = pred_mask_final.astype(np.uint8)
                        
                        if gt_mask_binary.sum() > 0:
                            dice, iou = get_metrics(pred_mask_binary, gt_mask_binary)
                            img_name = os.path.basename(image_file)
                            log_prefix = f"[Process {rank}]" if is_gpt_mode else f"[GPU {rank}]"
                            print(f"{log_prefix} {img_name} - Dice: {dice:.4f}, IoU: {iou:.4f}")
                            
                            # Get per-round IoU directly from inference_record (real-time computed)
                            per_round_metrics = inference_record.get('per_round_metrics', [])
                            num_clicks = len(inference_record.get('clicks_history', []))
                            
                            # Save to local metrics dict
                            mask_name = each_item['mask_file']
                            local_metrics[mask_name] = {
                                'dice': float(dice),
                                'IoU': float(iou),
                                'exp': target_description,
                                'num_clicks': num_clicks,
                                'per_round_metrics': per_round_metrics
                            }
                    except Exception as metric_error:
                        log_prefix = f"[Process {rank}]" if is_gpt_mode else f"[GPU {rank}]"
                        print(f"{log_prefix} Warning: Failed to compute metrics: {metric_error}")
                    
                    processed_count += 1
                    
                except Exception as mask_save_error:
                    log_prefix = f"[Process {rank}]" if is_gpt_mode else f"[GPU {rank}]"
                    print(f"{log_prefix} Error: Failed to save mask: {mask_save_error}")
                    failed_count += 1
                    continue
                    
            except Exception as e:
                log_prefix = f"[Process {rank}]" if is_gpt_mode else f"[GPU {rank}]"
                print(f"{log_prefix} Error processing sample: {e}")
                import traceback
                traceback.print_exc()
                failed_count += 1
                continue
        
        # Save stats and metrics for this process
        local_results = {
            'processed': processed_count,
            'failed': failed_count,
            'total': len(annotations_chunk),
            'metrics': local_metrics  # Metrics for each image
        }
        return_dict[rank] = local_results
        
        log_prefix = f"[Process {rank}]" if is_gpt_mode else f"[Process {rank} @ GPU {gpu_id}]"
        print(f"{log_prefix} Process completed! Success: {processed_count}, Failed: {failed_count}")
        
        # Clean up resources
        if hasattr(grounding_model, 'release_resources'):
            grounding_model.release_resources()
        if hasattr(segmentation_model, 'release_resources'):
            segmentation_model.release_resources()
        
        if not is_gpt_mode:
            torch.cuda.empty_cache()
        
    except Exception as e:
        log_prefix = f"[Process {rank}]" if is_gpt_mode else f"[Process {rank} @ GPU {gpu_id}]"
        print(f"{log_prefix} Process terminated with exception: {e}")
        import traceback
        traceback.print_exc()
        return_dict[rank] = {'processed': 0, 'failed': 0, 'total': 0, 'error': str(e)}


def split_data(annotations, total_processes):
    """
    Split dataset into total_processes chunks
    Args:
        annotations: Full annotations list
        total_processes: Total process count (n_gpus * processes_per_gpu)
    Returns:
        chunks: Split data list
    """
    chunk_size = len(annotations) // total_processes
    chunks = []
    
    for i in range(total_processes):
        start_idx = i * chunk_size
        if i == total_processes - 1:
            # Last process handles all remaining data
            end_idx = len(annotations)
        else:
            end_idx = start_idx + chunk_size
        
        chunks.append(annotations[start_idx:end_idx])
    
    return chunks


def main():
    """Main function"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Multi-GPU multi-process inference script')
    parser.add_argument('--dataset_name', type=str, default='BTCV',
                        help='Dataset name (BTCV, KiTS2023, Flare22, NeoPolyp, BreastUS, etc.)')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split (train, test, val)')
    parser.add_argument('--model_path', type=str, 
                        default='Qwen3/Qwen3-VL-8B-Instruct',
                        help='Grounding model path or model type (path or qwen/gpt, etc.)')
    parser.add_argument('--save_intermediate', type=str, default='false',
                        help='Whether to save intermediate results (true/false)')
    parser.add_argument('--n_clicks', type=int, default=3,
                        help='Maximum number of clicks')
    parser.add_argument('--n_gpus', type=int, default=8,
                        help='Number of GPUs to use')
    parser.add_argument('--processes_per_gpu', type=int, default=1,
                        help='Processes per GPU (for large-memory GPUs, e.g., 96GB can be set to 2)')
    parser.add_argument('--data_root', type=str,
                        default='your/data/root',
                        help='Dataset root directory')
    parser.add_argument('--output_dir', type=str,
                        default='./intermediate_results',
                        help='Directory to save intermediate results and inference records')
    parser.add_argument('--results_dir', type=str,
                        default='./results',
                        help='Directory to save final aggregated results')
    parser.add_argument('--resize_resolution', type=int, default=512,
                        help='Input image resolution; set to 0 to disable resize')
    parser.add_argument('--seg_model', type=str, default='imisnet',
                        help='Segmentation model type (sam, medsam, imisnet)')
    parser.add_argument('--seg-checkpoint', type=str,
                        default='your/segmentation/checkpoint/path',
                        help='Segmentation checkpoint path (sam/medsam/imisnet)')
    parser.add_argument('--seg-config', type=str,
                        default=None,
                        help='Segmentation config path (sam/medsam)')
    parser.add_argument('--use_fp16', type=str, default='true',
                        help='Use FP16/BF16 mixed precision (true/false), can significantly speed up and reduce VRAM')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch inference size (currently only batch_size=1; future versions will support larger batch)')
    # Optional GPT parameters (used in GPT mode)
    parser.add_argument('--gpt_api_key', type=str, default=None,
                        help='GPT API Key; if not provided, use environment variable OPENAI_API_KEY')
    parser.add_argument('--gpt_api_base', type=str, default=None,
                        help='GPT API Base URL, e.g., https://api.openai.com/v1 or other proxy')
    parser.add_argument('--gpt_model', type=str, default='gpt-4o',
                        help='GPT model name, e.g., gpt-4o, gpt-4-turbo, gpt-4o-mini')
    
    cmd_args = parser.parse_args()
    
    print("=" * 80)
    print("Multi-GPU multi-process inference script")
    print("=" * 80)
    
    # Pre-detect mode (based on model_path)
    is_gpt_mode = cmd_args.model_path and ('gpt' in cmd_args.model_path.lower())
    
    # Check CUDA availability (can be skipped in GPT mode)
    if not is_gpt_mode:
        if not torch.cuda.is_available():
            print("Error: CUDA not available, cannot use GPU")
            return
        
        available_gpus = torch.cuda.device_count()
        print(f"\nDetected {available_gpus} available GPUs")
        for i in range(available_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print(f"\n✨ GPT API mode - using serial calls, no GPU needed")
        available_gpus = 0  # GPT mode does not need GPU
    
    # Configure with command-line arguments
    DATASET_NAME = cmd_args.dataset_name
    SPLIT = cmd_args.split
    
    if not is_gpt_mode:
        N_GPUS = min(cmd_args.n_gpus, available_gpus)  # GPUs to use, not exceeding available count
        PROCESSES_PER_GPU = cmd_args.processes_per_gpu  # Processes per GPU
        TOTAL_PROCESSES = N_GPUS * PROCESSES_PER_GPU  # Total process count
        
        if N_GPUS < 1:
            print("Error: No available GPU")
            return
        
        print(f"\nWill use {N_GPUS} GPUs for inference")
        print(f"Processes per GPU: {PROCESSES_PER_GPU}")
        print(f"Total parallel processes: {TOTAL_PROCESSES}")
    else:
        # GPT mode: serial execution
        N_GPUS = 1
        PROCESSES_PER_GPU = 1
        TOTAL_PROCESSES = 1
    
    # Create configuration
    args = InferenceArgs()
    
    # Set model path or type
    # If model_path looks like a path (contains '/'), set as model path
    # Otherwise treat it as grounding_model type
    if '/' in cmd_args.model_path or os.path.exists(cmd_args.model_path):
        args.model = cmd_args.model_path
        args.grounding_model = "qwen"  # Default type when using checkpoint
        print(f"Using model checkpoint: {cmd_args.model_path}")
    else:
        args.grounding_model = cmd_args.model_path
        print(f"Using model type: {cmd_args.model_path}")

    # Pass GPT config to args (effective only when model type is gpt)
    is_gpt_mode = isinstance(args.grounding_model, str) and 'gpt' in args.grounding_model.lower()
    if is_gpt_mode:
        args.gpt_api_key = cmd_args.gpt_api_key
        args.gpt_api_base = cmd_args.gpt_api_base
        args.gpt_model = cmd_args.gpt_model
    
    args.max_history_length = 5
    args.use_history = True
    args.reset_history_per_image = True
    
    args.n_clicks = cmd_args.n_clicks
    args.use_previous_mask = True
    
    # Convert save_intermediate parameter
    args.save_intermediate = cmd_args.save_intermediate.lower() in ('true', '1', 'yes', 'y')
    
    # Convert use_fp16 parameter
    args.use_fp16 = cmd_args.use_fp16.lower() in ('true', '1', 'yes', 'y')
    
    # Set batch_size
    args.batch_size = cmd_args.batch_size
    if args.batch_size > 1:
        print(f"\nWarning: Current batch_size={args.batch_size}, but batch inference is still under development")
        print("      Using batch_size=1 for inference")
        args.batch_size = 1
    
    # Set resize resolution
    args.resize_resolution = cmd_args.resize_resolution if cmd_args.resize_resolution > 0 else None
    
    # Set segmentation model parameters
    args.seg_model = cmd_args.seg_model
    args.seg_checkpoint = cmd_args.seg_checkpoint
    args.seg_config = cmd_args.seg_config
    
    args.split = SPLIT
    args.dataset_name = DATASET_NAME
    args.val_folder = os.path.join(cmd_args.data_root, DATASET_NAME)
    args.val_json = '{}.json'.format(SPLIT)
    
    # Set output directories
    args.output_dir = cmd_args.output_dir
    args.results_dir = cmd_args.results_dir
    
    print("\nConfiguration:")
    print(f"  Dataset: {DATASET_NAME}")
    print(f"  Split: {SPLIT}")
    print(f"  Data folder: {args.val_folder}")
    
    # Detect GPT mode (before configuring GPUs)
    is_gpt_mode = isinstance(args.grounding_model, str) and 'gpt' in args.grounding_model.lower()
    
    if is_gpt_mode:
        print(f"\n✨ GPT mode detected, using serial API calls")
        print("  - No GPU multi-processing")
        print("  - Call model via OpenAI API")
        N_GPUS = 1
        PROCESSES_PER_GPU = 1
        TOTAL_PROCESSES = 1
    else:
        print(f"  GPU count: {N_GPUS}")
        print(f"  Processes per GPU: {PROCESSES_PER_GPU}")
        print(f"  Total parallel processes: {TOTAL_PROCESSES}")
    
    print(f"  Grounding model: {args.grounding_model}")
    print(f"  Model path: {getattr(args, 'model', 'default')}")
    print(f"  Max clicks: {args.n_clicks}")
    print(f"  Max history length: {args.max_history_length}")
    print(f"  Segmentation model: {args.seg_model}")
    print(f"  Save intermediate: {args.save_intermediate}")
    print(f"  Image resolution: {args.resize_resolution if args.resize_resolution else 'Original size'}")
    print(f"  Intermediate output dir: {args.output_dir}")
    print(f"  Results output dir: {args.results_dir}")
    print(f"  Use FP16 mixed precision: {args.use_fp16}")
    print(f"  Batch size: {args.batch_size}")
    
    # Load dataset
    json_file = os.path.join(args.val_folder, '{}.json'.format(SPLIT))
    
    if not os.path.exists(json_file):
        print(f"Error: Dataset file not found: {json_file}")
        return
        
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    annotations = data['annotations']
    print(f"\nTotal samples: {len(annotations)}")
    
    # Split data by total processes
    data_chunks = split_data(annotations, TOTAL_PROCESSES)
    
    print("\nData split:")
    for i, chunk in enumerate(data_chunks):
        gpu_id = i // PROCESSES_PER_GPU
        process_id_on_gpu = i % PROCESSES_PER_GPU
        print(f"  Process {i} (GPU {gpu_id}, Process #{process_id_on_gpu}): {len(chunk)} samples")
    
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Create shared dict for collecting results
    manager = mp.Manager()
    return_dict = manager.dict()
    
    # Create processes
    processes = []
    print("\nStarting multi-process inference...")
    start_time = time.time()
    
    for rank in range(TOTAL_PROCESSES):
        p = mp.Process(
            target=worker_process,
            args=(rank, N_GPUS, PROCESSES_PER_GPU, data_chunks[rank], args, return_dict)
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Aggregate results
    print("\n" + "=" * 80)
    print("Inference complete!")
    print("=" * 80)
    
    total_processed = 0
    total_failed = 0
    all_metrics = {}  # Aggregate metrics from all processes
    
    print("\nPer-process statistics:")
    for rank in range(TOTAL_PROCESSES):
        if rank in return_dict:
            result = return_dict[rank]
            gpu_id = rank // PROCESSES_PER_GPU
            process_id_on_gpu = rank % PROCESSES_PER_GPU
            if 'error' in result:
                print(f"  Process {rank} (GPU {gpu_id}, Process #{process_id_on_gpu}): Error - {result['error']}")
            else:
                print(f"  Process {rank} (GPU {gpu_id}, Process #{process_id_on_gpu}): Success {result['processed']}, "
                      f"Failed {result['failed']}, "
                      f"Total {result['total']}")
                total_processed += result['processed']
                total_failed += result['failed']
                
                # Merge per-process metrics
                if 'metrics' in result:
                    all_metrics.update(result['metrics'])
        else:
            print(f"  Process {rank}: No result")
    
    print(f"\nOverall statistics:")
    print(f"  Successfully processed: {total_processed} samples")
    print(f"  Failed samples: {total_failed}")
    print(f"  Total: {len(annotations)}")
    print(f"  Success rate: {total_processed/len(annotations)*100:.1f}%")
    print(f"  Total time: {total_time:.2f} s")
    print(f"  Avg per sample: {total_time/len(annotations):.2f} s")
    
    # Compute metrics
    if total_processed > 0 and len(all_metrics) > 0:
        print("\nComputing evaluation metrics...")
        try:
            # Compute mean using aggregated metrics
            mean_metrics = get_mean(all_metrics)
            
            # Save results using configured results_dir
            results_dir = args.results_dir
            os.makedirs(results_dir, exist_ok=True)
            
            # Build final results with mean first
            final_results = {'mean': mean_metrics}
            final_results.update(all_metrics)
            
            results_json = os.path.join(results_dir, 'results.json')
            with open(results_json, 'w') as f:
                json.dump(final_results, f, indent=4)
            
            print("✅ Metrics computed")
            print("\nFinal evaluation results:")
            print(f"  Mean Class Dice: {mean_metrics.get('mean_class_dice', 0):.4f}")
            print(f"  Mean Class IoU:  {mean_metrics.get('mean_class_iou', 0):.4f}")
            print(f"  Mean Dice:       {mean_metrics.get('mean_dice', 0):.4f}")
            print(f"  Mean IoU:        {mean_metrics.get('mean_iou', 0):.4f}")
            
            # Added: analyze click counts and per-round IoU
            print("\nAnalyzing click counts and per-round IoU...")
            
            click_counts = []
            per_round_ious = {}  # {round_id: [iou_list]}
            per_round_dices = {}  # {round_id: [dice_list]}
            
            # Collect click counts and per-round IoU for all samples
            for mask_name, metrics in all_metrics.items():
                if 'num_clicks' in metrics:
                    click_counts.append(metrics['num_clicks'])
                
                # If per-round metrics exist (real-time computed)
                if 'per_round_metrics' in metrics:
                    for round_metric in metrics['per_round_metrics']:
                        round_id = round_metric['round']
                        if round_id not in per_round_ious:
                            per_round_ious[round_id] = []
                            per_round_dices[round_id] = []
                        per_round_ious[round_id].append(round_metric.get('iou', 0))
                        per_round_dices[round_id].append(round_metric.get('dice', 0))
            
            # Click count statistics
            if click_counts:
                from collections import Counter
                click_counter = Counter(click_counts)
                
                print("\n" + "=" * 80)
                print("Click count analysis:")
                print("=" * 80)
                print(f"Total samples: {len(click_counts)}")
                print(f"Average clicks: {sum(click_counts) / len(click_counts):.2f}")
                print(f"Minimum clicks: {min(click_counts)}")
                print(f"Maximum clicks: {max(click_counts)}")
                
                print("\nClick count frequency distribution:")
                for clicks, count in sorted(click_counter.items()):
                    percentage = count / len(click_counts) * 100
                    print(f"  {clicks} clicks: {count} samples ({percentage:.1f}%)")
                
                # Save click count statistics to results
                click_stats = {
                    'total_samples': len(click_counts),
                    'mean_clicks': float(sum(click_counts) / len(click_counts)),
                    'min_clicks': int(min(click_counts)),
                    'max_clicks': int(max(click_counts)),
                    'click_frequency': {str(k): int(v) for k, v in sorted(click_counter.items())}
                }
                final_results['click_statistics'] = click_stats
                
                print(f"\n✅ Click count statistics added")
            else:
                print(f"\n⚠️  Warning: No click count data collected")
            
            # Per-round IoU statistics
            if per_round_ious:
                print("\n" + "=" * 80)
                print("Per-round IoU statistics:")
                print("=" * 80)
                print(f"{'Round':<6} {'Mean IoU':<10} {'Mean Dice':<10} {'Samples':<8}")
                print("-" * 40)
                
                per_round_stats = {}
                for round_id in sorted(per_round_ious.keys()):
                    ious = per_round_ious[round_id]
                    dices = per_round_dices[round_id]
                    avg_iou = sum(ious) / len(ious)
                    avg_dice = sum(dices) / len(dices)
                    print(f"{round_id:<6} {avg_iou:<10.4f} {avg_dice:<10.4f} {len(ious):<8}")
                    
                    per_round_stats[str(round_id)] = {
                        'mean_iou': float(avg_iou),
                        'mean_dice': float(avg_dice),
                        'num_samples': len(ious)
                    }
                
                final_results['per_round_statistics'] = per_round_stats
                print(f"\n✅ Per-round IoU statistics added")
            else:
                print(f"\n⚠️  Warning: No per-round IoU data collected")
            
            # Save full results.json (including statistics)
            try:
                with open(results_json, 'w') as f:
                    json.dump(final_results, f, indent=4)
                
                print(f"\n✅ All results saved to {results_json}")
                print(f"   File contains:")
                print(f"   - mean: mean metrics")
                if 'click_statistics' in final_results:
                    print(f"   - click_statistics: click count statistics")
                if 'per_round_statistics' in final_results:
                    print(f"   - per_round_statistics: per-round IoU statistics")
                print(f"   - {len(all_metrics)} samples of detailed metrics")
                
            except Exception as save_error:
                print(f"❌ Error: Failed to save results.json: {save_error}")
                import traceback
                traceback.print_exc()
                    
        except Exception as metrics_error:
            print(f"Error: Metric computation failed: {metrics_error}")
            import traceback
            traceback.print_exc()
    else:
        print("\nWarning: No samples processed successfully, skip metric computation")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
