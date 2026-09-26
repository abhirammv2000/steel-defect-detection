"""Inference service for the trained steel defect segmentation model.

Three response fields, each its own capability:
  - segmentation: per-class predicted mask, returned as area coverage and
    confidence rather than the full-resolution mask, to keep responses small
  - anomaly detection: defect_present, derived from whether any predicted
    mask is non-empty
  - objective grading: a severity score computed from predicted defect area
    and class mix, see app/grading.py for the formula
"""
import io
import numpy as np
import cv2
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
import segmentation_models_pytorch as smp

from app.dataset import NUM_CLASSES, TRAIN_RESIZE
from app.grading import compute_grade

MODEL_PATH="models/steel_defect_unet.pt"
PRED_THRESHOLD=0.5

app=FastAPI(title="Steel Defect Detection")
_model=None


def get_model():
    global _model
    if _model is None:
        model=smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=NUM_CLASSES)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        _model=model
    return _model


@app.get("/health")
def health():
    return {"status":"healthy"}


@app.post("/predict")
async def predict(file: UploadFile=File(...)):
    raw=await file.read()
    image_array=np.frombuffer(raw, dtype=np.uint8)
    image=cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized=cv2.resize(image, (TRAIN_RESIZE[1], TRAIN_RESIZE[0]))
    tensor=torch.from_numpy(resized.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0)

    model=get_model()
    with torch.no_grad():
        logits=model(tensor)
        probs=torch.sigmoid(logits)[0] # (NUM_CLASSES, H, W)

    masks=(probs>PRED_THRESHOLD).numpy()
    total_pixels=masks.shape[1]*masks.shape[2]

    per_class=[]
    for class_id in range(NUM_CLASSES):
        area_fraction=float(masks[class_id].sum())/total_pixels
        per_class.append({
            "class_id": class_id+1,
            "defect_present": bool(masks[class_id].any()),
            "area_fraction": round(area_fraction, 5),
            "max_confidence": round(float(probs[class_id].max()), 4),
        })

    defect_present=any(c["defect_present"] for c in per_class)
    grade=compute_grade(per_class)

    return {
        "defect_present": defect_present, # anomaly detection
        "classes": per_class, # segmentation summary
        "grade": grade, # objective grading
    }
