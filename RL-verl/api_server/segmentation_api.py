"""
Unified IMISNet / MedSAM2 FastAPI Web Backend
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
from typing import Optional, List, Dict, Tuple
import torch
import numpy as np
from PIL import Image
import io
import base64
import sys
import os
from pathlib import Path
import json
import uuid
from datetime import datetime, timedelta

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
rl_verl_root = project_root / "RL-verl"
sam2_root = project_root / "third_party" / "sam2"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(rl_verl_root))
sys.path.insert(0, str(sam2_root))

# IMISNet imports
from third_party.segment_anything import sam_model_registry
from third_party.segment_anything.predictor import IMISPredictor
from third_party.segment_anything.model import IMISNet
from argparse import Namespace

# MedSAM2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Initialize FastAPI app
app = FastAPI(
    title="Unified IMISNet/MedSAM2 API",
    description="Medical Image Segmentation with IMISNet or MedSAM2",
    version="1.1.0"
)

# Global variables
predictor = None
device = None
model_type = None

# Session storage
sessions: Dict[str, dict] = {}


class PredictResponse(BaseModel):
    """Prediction response model."""
    masks: List[str]  # List of Base64-encoded mask images
    category_pred: List[str]
    message: str


class SessionResponse(BaseModel):
    """Session response model."""
    session_id: str
    message: str
    image_size: Optional[tuple] = None


class InteractiveResponse(BaseModel):
    """Interactive segmentation response."""
    session_id: str
    masks: List[str]
    category_pred: List[str]
    round_number: int
    total_clicks: int
    message: str


def cleanup_old_sessions(max_age_hours: int = 24) -> int:
    """Clean sessions not accessed within the specified time."""
    current_time = datetime.now()
    expired_sessions = []

    for session_id, session_data in sessions.items():
        age = current_time - session_data["last_accessed"]
        if age > timedelta(hours=max_age_hours):
            expired_sessions.append(session_id)

    for session_id in expired_sessions:
        del sessions[session_id]

    return len(expired_sessions)


def image_to_base64(image_array: np.ndarray) -> str:
    """Convert a numpy array to a base64-encoded PNG image."""
    if image_array.max() <= 1.0:
        image_array = (image_array * 255).astype(np.uint8)
    else:
        image_array = image_array.astype(np.uint8)

    image = Image.fromarray(image_array)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def _parse_json_array(value: Optional[str]) -> Optional[np.ndarray]:
    if value is None:
        return None
    return np.array(json.loads(value))


def _decode_logits(base64_str: str, shape: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    logits_bytes = base64.b64decode(base64_str)
    logits_np = np.frombuffer(logits_bytes, dtype=np.float32)
    if shape is None:
        shape = (1, 256, 256)
    expected = int(np.prod(shape))
    if logits_np.size != expected:
        raise ValueError(f"Logits size mismatch: expected {expected}, got {logits_np.size}")
    return logits_np.reshape(*shape)


def _predict(
    point_coords_np: Optional[np.ndarray],
    point_labels_np: Optional[np.ndarray],
    bbox_np: Optional[np.ndarray],
    text_list: Optional[List[str]],
    mask_input: Optional[np.ndarray],
    multimask_output: bool,
):
    """Unified prediction interface, returns masks, logits, category_pred."""
    if model_type == "imisnet":
        masks, logits, category_pred = predictor.predict(
            point_coords=point_coords_np,
            point_labels=point_labels_np,
            box=bbox_np,
            text=text_list,
            mask_input=mask_input,
            multimask_output=multimask_output,
        )
        return masks, logits, category_pred

    # MedSAM2
    masks, scores, logits = predictor.predict(
        point_coords=point_coords_np,
        point_labels=point_labels_np,
        box=bbox_np,
        mask_input=mask_input,
        multimask_output=multimask_output,
    )

    max_score_idx = int(np.argmax(scores))
    best_mask = masks[max_score_idx]
    best_logits = logits[[max_score_idx]]
    return [best_mask], best_logits, []


@app.on_event("startup")
async def startup_event():
    """Load the model at startup."""
    global predictor, device, model_type

    model_type = os.getenv("MODEL_TYPE", "imisnet").lower().strip()
    if model_type not in {"imisnet", "medsam2"}:
        raise ValueError("MODEL_TYPE supports only imisnet or medsam2")

    print(f"Loading model: {model_type}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if model_type == "imisnet":
        args = Namespace()
        args.image_size = int(os.getenv("IMISNET_IMAGE_SIZE", "1024"))
        args.sam_checkpoint = os.getenv("IMISNET_CHECKPOINT")
        category_weights = None

        if not os.path.exists(args.sam_checkpoint):
            raise FileNotFoundError(f"IMISNet checkpoint not found: {args.sam_checkpoint}")

        sam = sam_model_registry["vit_b"](args).to(device)
        imisnet = IMISNet(sam, test_mode=True, category_weights=category_weights).to(device)
        predictor = IMISPredictor(imisnet)
        print("IMISNet model loaded successfully!")
        return

    # MedSAM2
    sam_checkpoint = os.getenv("MEDSAM2_CHECKPOINT")
    model_config = os.getenv(
        "MEDSAM2_CONFIG",
        "configs/sam2.1/sam2.1_hiera_t.yaml",
    )

    if not os.path.exists(sam_checkpoint):
        raise FileNotFoundError(
            f"MedSAM2 checkpoint not found: {sam_checkpoint}\n"
            "Please set MEDSAM2_CHECKPOINT to the correct model path"
        )

    sam2_model = build_sam2(model_config, sam_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    print(f"MedSAM2 model loaded successfully! Model path: {sam_checkpoint}")


@app.get("/")
async def root():
    return {
        "message": "Unified IMISNet/MedSAM2 API is running",
        "version": "1.1.0",
        "model_type": model_type,
        "endpoints": {
            "/predict": "POST - Upload image and perform segmentation prediction",
            "/predict_with_state": "POST - Iterative prediction using previous mask state",
            "/session/create": "POST - Create a new segmentation session",
            "/session/add_click": "POST - Add a click for segmentation",
            "/session/add_bbox": "POST - Add a bounding box for segmentation",
            "/session/segment": "POST - Segment with specified prompts",
            "/session/{session_id}/status": "GET - Query session status",
            "/session/{session_id}": "DELETE - Delete session",
            "/sessions": "GET - List all sessions",
            "/health": "GET - Check service health",
        },
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "model_type": model_type,
        "device": str(device),
        "active_sessions": len(sessions),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(
    image: UploadFile = File(..., description="Input image file (PNG, JPG, etc.)"),
    point_coords: Optional[str] = Form(None, description="Point coordinates JSON: [[x1,y1],[x2,y2]]"),
    point_labels: Optional[str] = Form(None, description="Point labels JSON: [1,0,1]"),
    bbox: Optional[str] = Form(None, description="Bounding box JSON: [x1,y1,x2,y2]"),
    text: Optional[str] = Form(None, description="Text prompt JSON: ['kidney_left']"),
    multimask_output: bool = Form(False, description="Whether to output multiple masks"),
):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        image_bytes = await image.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image_pil)

        predictor.set_image(image_np)

        point_coords_np = _parse_json_array(point_coords)
        point_labels_np = _parse_json_array(point_labels)
        bbox_np = _parse_json_array(bbox)
        text_list = json.loads(text) if text is not None else None

        masks, logits, category_pred = _predict(
            point_coords_np=point_coords_np,
            point_labels_np=point_labels_np,
            bbox_np=bbox_np,
            text_list=text_list,
            mask_input=None,
            multimask_output=multimask_output,
        )

        masks_base64 = [image_to_base64(mask.squeeze()) for mask in masks]

        return PredictResponse(
            masks=masks_base64,
            category_pred=category_pred,
            message="Prediction succeeded",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_with_state")
async def predict_with_state(
    image: UploadFile = File(...),
    previous_logits: str = Form(..., description="Previous logits (base64-encoded)"),
    previous_logits_shape: Optional[str] = Form(None, description="Logits shape JSON: [1,256,256]"),
    point_coords: Optional[str] = Form(None),
    point_labels: Optional[str] = Form(None),
    bbox: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    multimask_output: bool = Form(False),
):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        image_bytes = await image.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image_pil)

        predictor.set_image(image_np)

        shape = json.loads(previous_logits_shape) if previous_logits_shape is not None else None
        logits_np = _decode_logits(previous_logits, tuple(shape) if shape is not None else None)

        point_coords_np = _parse_json_array(point_coords)
        point_labels_np = _parse_json_array(point_labels)
        bbox_np = _parse_json_array(bbox)
        text_list = json.loads(text) if text is not None else None

        masks, logits, category_pred = _predict(
            point_coords_np=point_coords_np,
            point_labels_np=point_labels_np,
            bbox_np=bbox_np,
            text_list=text_list,
            mask_input=logits_np,
            multimask_output=multimask_output,
        )

        masks_base64 = [image_to_base64(mask.squeeze()) for mask in masks]
        logits_base64 = base64.b64encode(logits.tobytes()).decode()

        return {
            "masks": masks_base64,
            "category_pred": category_pred,
            "logits": logits_base64,
            "logits_shape": list(logits.shape),
            "message": "Prediction succeeded",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/session/create", response_model=SessionResponse)
async def create_session(
    image: UploadFile = File(..., description="Input image file"),
    session_id: Optional[str] = Form(None, description="Optional session ID; auto-generate if not provided"),
):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        if session_id is None:
            session_id = str(uuid.uuid4())

        image_bytes = await image.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image_pil)

        predictor.set_image(image_np)

        sessions[session_id] = {
            "image": image_pil,
            "image_np": image_np,
            "clicks": [],
            "logits": None,
            "last_mask": None,
            "created_at": datetime.now(),
            "last_accessed": datetime.now(),
            "round_number": 0,
        }

        cleanup_old_sessions()

        return SessionResponse(
            session_id=session_id,
            message="Session created successfully",
            image_size=(image_pil.width, image_pil.height),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@app.post("/session/add_click", response_model=InteractiveResponse)
async def add_click_and_segment(
    session_id: str = Form(..., description="Session ID"),
    point_x: int = Form(..., description="x coordinate of the click point"),
    point_y: int = Form(..., description="y coordinate of the click point"),
    is_positive: bool = Form(True, description="Whether positive (True=foreground, False=background)"),
    multimask_output: bool = Form(False, description="Whether to output multiple masks"),
):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    try:
        session = sessions[session_id]
        session["last_accessed"] = datetime.now()
        session["round_number"] += 1

        predictor.set_image(session["image_np"])

        label = 1 if is_positive else 0
        session["clicks"].append((point_x, point_y, label))

        point_coords_np = np.array([[point_x, point_y]])
        point_labels_np = np.array([label])

        masks, logits, category_pred = _predict(
            point_coords_np=point_coords_np,
            point_labels_np=point_labels_np,
            bbox_np=None,
            text_list=None,
            mask_input=session["logits"],
            multimask_output=multimask_output,
        )

        session["logits"] = logits
        session["last_mask"] = masks[0] if len(masks) > 0 else None

        masks_base64 = [image_to_base64(mask.squeeze()) for mask in masks]

        return InteractiveResponse(
            session_id=session_id,
            masks=masks_base64,
            category_pred=category_pred,
            round_number=session["round_number"],
            total_clicks=len(session["clicks"]),
            message=f"Segmentation round {session['round_number']} completed",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


@app.post("/session/add_bbox", response_model=InteractiveResponse)
async def add_bbox_and_segment(
    session_id: str = Form(..., description="Session ID"),
    bbox: str = Form(..., description="Bounding box JSON: [x1,y1,x2,y2]"),
    multimask_output: bool = Form(False, description="Whether to output multiple masks"),
):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    try:
        session = sessions[session_id]
        session["last_accessed"] = datetime.now()
        session["round_number"] += 1

        predictor.set_image(session["image_np"])

        bbox_list = json.loads(bbox)
        bbox_np = np.array(bbox_list)

        session["clicks"].append(f"bbox:{bbox}")

        masks, logits, category_pred = _predict(
            point_coords_np=None,
            point_labels_np=None,
            bbox_np=bbox_np,
            text_list=None,
            mask_input=session["logits"],
            multimask_output=multimask_output,
        )

        session["logits"] = logits
        session["last_mask"] = masks[0] if len(masks) > 0 else None

        masks_base64 = [image_to_base64(mask.squeeze()) for mask in masks]

        return InteractiveResponse(
            session_id=session_id,
            masks=masks_base64,
            category_pred=category_pred,
            round_number=session["round_number"],
            total_clicks=len(session["clicks"]),
            message=f"Segmentation round {session['round_number']} completed (bbox initialization)",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


@app.post("/session/segment", response_model=InteractiveResponse)
async def segment_with_prompts(
    session_id: str = Form(..., description="Session ID"),
    point_coords: Optional[str] = Form(None, description="Point coordinates JSON: [[x1,y1],[x2,y2]]"),
    point_labels: Optional[str] = Form(None, description="Point labels JSON: [1,0,1]"),
    bbox: Optional[str] = Form(None, description="Bounding box JSON: [x1,y1,x2,y2]"),
    use_previous_mask: bool = Form(True, description="Whether to use previous round mask"),
    multimask_output: bool = Form(False, description="Whether to output multiple masks"),
):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    try:
        session = sessions[session_id]
        session["last_accessed"] = datetime.now()
        session["round_number"] += 1

        predictor.set_image(session["image_np"])

        point_coords_np = _parse_json_array(point_coords)
        point_labels_np = _parse_json_array(point_labels)
        bbox_np = _parse_json_array(bbox)

        mask_input = session["logits"] if use_previous_mask else None

        masks, logits, category_pred = _predict(
            point_coords_np=point_coords_np,
            point_labels_np=point_labels_np,
            bbox_np=bbox_np,
            text_list=None,
            mask_input=mask_input,
            multimask_output=multimask_output,
        )

        session["logits"] = logits
        session["last_mask"] = masks[0] if len(masks) > 0 else None

        masks_base64 = [image_to_base64(mask.squeeze()) for mask in masks]

        return InteractiveResponse(
            session_id=session_id,
            masks=masks_base64,
            category_pred=category_pred,
            round_number=session["round_number"],
            total_clicks=len(session.get("clicks", [])),
            message=f"Segmentation round {session['round_number']} completed",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


@app.get("/session/{session_id}/status")
async def get_session_status(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    session = sessions[session_id]
    return {
        "session_id": session_id,
        "round_number": session["round_number"],
        "total_clicks": len(session["clicks"]),
        "clicks_history": session["clicks"],
        "has_mask": session["last_mask"] is not None,
        "created_at": session["created_at"].isoformat(),
        "last_accessed": session["last_accessed"].isoformat(),
        "image_size": session["image"].size,
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    del sessions[session_id]
    return {
        "session_id": session_id,
        "message": "Session deleted",
        "remaining_sessions": len(sessions),
    }


@app.get("/sessions")
async def list_sessions():
    session_list = []
    for session_id, session_data in sessions.items():
        session_list.append(
            {
                "session_id": session_id,
                "round_number": session_data["round_number"],
                "total_clicks": len(session_data["clicks"]),
                "created_at": session_data["created_at"].isoformat(),
                "last_accessed": session_data["last_accessed"].isoformat(),
            }
        )

    return {"total_sessions": len(sessions), "sessions": session_list}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8265")),
        log_level="info",
    )
