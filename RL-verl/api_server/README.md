# Segmentation API (IMISNet / MedSAM2)

This directory contains a unified FastAPI server for interactive medical image segmentation using either IMISNet or MedSAM2.

Files
- `segmentation_api.py`: Unified FastAPI application supporting `imisnet` and `medsam2`.
- `run_api.sh`: Launcher script that reads environment variables to decide which model and checkpoints to load.

Quick start
1. Install dependencies (if not already installed):

```bash
pip install -r RL-verl/api_server/requirements.txt
```

2. Make the launcher executable (one-time):

```bash
chmod +x RL-verl/api_server/run_api.sh
```

3. Run the server

- Start with the default model (IMISNet):

```bash
MODEL_TYPE=imisnet RL-verl/api_server/run_api.sh
```

- Start with MedSAM2 and override checkpoint/config paths:

```bash
MODEL_TYPE=medsam2 MEDSAM2_CHECKPOINT=/path/to/MedSAM2.pt MEDSAM2_CONFIG=configs/sam2.1/sam2.1_hiera_t.yaml RL-verl/api_server/run_api.sh
```

- Start with IMISNet and override checkpoint path:

```bash
MODEL_TYPE=imisnet IMISNET_CHECKPOINT=/path/to/IMISNet-B.pth IMISNET_CATEGORY_WEIGHTS=third_party/segment_anything/dataloaders/categories_weight.pkl RL-verl/api_server/run_api.sh
```

Run in background (write logs to file):

```bash
MODEL_TYPE=medsam2 MEDSAM2_CHECKPOINT=/path/to/MedSAM2.pt nohup RL-verl/api_server/run_api.sh > /tmp/seg_api.log 2>&1 &
```

Health check

```bash
curl http://localhost:8265/health
```

Integration notes
- The launcher reads environment variables to determine which model and checkpoint to use. To centralize configuration, set checkpoint variables only in `RL-verl/api_server/run_api.sh` or in your environment/CI, and do not duplicate paths in training scripts.
- The server port can be overridden with the `PORT` environment variable (default: 8265).

Next steps
- If you want the training script `recipe/medsam_agent/run.sh` to automatically start the API in the background and wait until it is ready, I can add that behavior.
- If you prefer a systemd unit or other supervisor for long-running deployment, I can provide an example unit file.

If you want additional changes, tell me which integration behaviour you prefer.