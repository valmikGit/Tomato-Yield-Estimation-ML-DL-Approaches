import streamlit as st
import torch
import json
import numpy as np
import cv2
from pathlib import Path
from skimage import io
import tempfile
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --------------------
# Load Feature Extract Function
# --------------------
# Use SAME function you used during training
mobilenet_weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
mobilenet = mobilenet_v3_large(weights=mobilenet_weights).to(DEVICE)
mobilenet.eval()
# remove the classifier head: keep features and adaptive pooling
pool = nn.AdaptiveAvgPool2d((1,1)).to(DEVICE)


def extract_features_from_crop(crop_path):
    img = io.imread(crop_path).astype(np.float32) / 255.0  # HWC [0,1]
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    if img.shape[2] == 4:
        img = img[..., :3]
    # apply mobilenet preprocessing
    preprocess = mobilenet_weights.transforms()
    # use PIL image via skimage -> convert to uint8 array then to PIL
    from PIL import Image
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    inp = preprocess(pil_img).unsqueeze(0).to(DEVICE)  # shape (1,3,H,W)
    with torch.no_grad():
        feat_map = mobilenet.features(inp)  # shape (1, C, h, w)
        pooled = pool(feat_map)  # (1, C, 1, 1)
        vec = pooled.view(1, -1)  # (1, C)
        # to scalar alpha, take mean of feature vector
        alpha = float(vec.mean().cpu().numpy())
    H, W = img.shape[:2]
    area = float(H * W)
    aspect_ratio = float(W / H) if H != 0 else 0.0
    avg_pixel_intensity = float(img.mean())
    return alpha, area, aspect_ratio, avg_pixel_intensity


# --------------------
# Load Models
# --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_OUT_DIR = "vaibhav_models"

# Load detector

detector = fasterrcnn_resnet50_fpn(weights=None)

# Replace predictor with SAME number of classes used during training
num_classes = 2  # background + tomato
in_features = detector.roi_heads.box_predictor.cls_score.in_features
detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load weights
detector.load_state_dict(torch.load(
    f"{MODEL_OUT_DIR}/fasterrcnn_resnet50_fpn_detector.pth",
    map_location=DEVICE
))
detector.to(DEVICE).eval()


class WeightMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(-1)


mlp = WeightMLP()
mlp.load_state_dict(torch.load(
    f"{MODEL_OUT_DIR}/weight_mlp.pth",
    map_location=DEVICE
))
mlp.to(DEVICE).eval()


# Load normalization info (if required later)
with open(f"{MODEL_OUT_DIR}/normalization_info.json", "r") as f:
    norm_info = json.load(f)

IMG_SIZE = (640, 640)


# -----------------------------
# INFERENCE FUNCTION
# -----------------------------
def infer_weight(image_path, detector_model, mlp_model):
    detector_model.eval()
    mlp_model.eval()

    img = io.imread(image_path)

    # Ensure 3 channels
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:
        img = img[..., :3]

    resized = cv2.resize(img, IMG_SIZE)
    inp = torch.from_numpy(np.transpose(resized.astype(np.float32) / 255.0, (2,0,1))).float().to(DEVICE)

    with torch.no_grad():
        outs = detector_model([inp])

    boxes = outs[0].get("boxes", torch.empty((0,4))).cpu().numpy()
    scores = outs[0].get("scores", torch.empty((0,))).cpu().numpy()

    # Choose detected tomato
    if len(boxes) == 0:
        crop = resized.astype(np.float32) / 255.0
    else:
        best_idx = int(np.argmax(scores))
        x1,y1,x2,y2 = boxes[best_idx].astype(int)
        H,W = resized.shape[:2]
        x1,y1,x2,y2 = max(0,x1), max(0,y1), min(W-1,x2), min(H-1,y2)
        crop = resized[y1:y2+1, x1:x2+1].astype(np.float32) / 255.0

    # Save crop temporarily
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        crop_path = tmp.name
    io.imsave(crop_path, (crop * 255).astype(np.uint8))

    # Extract features
    alpha, area, ar, api = extract_features_from_crop(crop_path)
    os.remove(crop_path)

    # MLP Input
    feat = torch.tensor([alpha, area, ar, api], dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_weight = mlp_model(feat).cpu().item()

    return pred_weight, resized, boxes, scores


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🍅 Tomato Weight Estimation – End-to-End Inference")
st.write("Upload an image of a tomato. The system will detect it, extract geometric features, and predict its weight.")

uploaded = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(uploaded.read())

    st.image(uploaded, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Running inference…"):
        pred_weight, resized_img, boxes, scores = infer_weight(tmp_path, detector, mlp)

    st.subheader("Predicted Weight")
    st.success(f"**{pred_weight:.2f} grams**")

    # Draw bounding box on image if detected
    if len(boxes) > 0:
        best_idx = int(np.argmax(scores))
        x1,y1,x2,y2 = boxes[best_idx].astype(int)

        disp = resized_img.copy()
        cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,0), 2)
        st.image(disp, caption="Detection Result")

    else:
        st.warning("⚠ No detection found, whole image was used.")

    os.remove(tmp_path)

