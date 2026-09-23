"""PyTorch model loading + Grad-CAM utilities."""

import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

IMG_SIZE = 224
DEVICE = torch.device("cpu")


class CassavaCNN(nn.Module):
    """Custom CNN — MODIFY to match your training architecture."""
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_model(architecture: str, num_classes: int):
    arch = architecture.lower()

    if arch == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        target_layers = [model.layer4[-1]]
    elif arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        target_layers = [model.layer4[-1]]
    elif arch == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        target_layers = [model.features[-1]]
    elif arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        target_layers = [model.features[-1]]
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        target_layers = [model.features[-1]]
    elif arch == "custom_cnn":
        model = CassavaCNN(num_classes=num_classes)
        target_layers = [model.features[-3]]
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return model, target_layers


def load_cassava_model(model_path: str, metadata_path: str = None):
    # ---- Load checkpoint ----
    checkpoint = torch.load(model_path, map_location=DEVICE)

    # ---- Unwrap nested checkpoint ----
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # ---- Strip 'module.' prefix if present ----
    state_dict = {
        (k.replace("module.", "", 1) if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    # ---- Auto-detect num_classes from checkpoint ----
    num_classes = state_dict["classifier.1.weight"].shape[0]

    # ---- Build MobileNetV2 ----
    model, target_layers = build_model("mobilenet_v2", num_classes)

    # ---- Load weights ----
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    # ---- Resolve class names + img_size ----
    class_names = None
    img_size = IMG_SIZE

    # Try metadata file first
    if metadata_path:
        try:
            with open(metadata_path) as f:
                meta = json.load(f)
            class_names = meta.get("class_names")
            img_size = meta.get("img_size", IMG_SIZE)
        except FileNotFoundError:
            pass

    # Fall back to checkpoint's own class_names
    if class_names is None and isinstance(checkpoint, dict):
        class_names = checkpoint.get("class_names")
        img_size = checkpoint.get("input_size", img_size)

    # Last resort
    if class_names is None:
        class_names = [
            "Cassava Bacterial Blight (CBB)",
            "Cassava Brown Streak Disease (CBSD)",
            "Cassava Green Mottle Disease (CGMD)",
            "Cassava Mosaic Disease (CMD)",
            "Healthy",
        ]

    return model, class_names, target_layers, img_size


def get_transform(img_size: int = IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def preprocess_image(pil_image, img_size: int = IMG_SIZE):
    return get_transform(img_size)(pil_image.convert("RGB")).unsqueeze(0).to(DEVICE)


@torch.no_grad()
def predict(model, batch):
    logits = model(batch)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    return idx, float(probs[idx]), probs


def generate_gradcam(model, target_layers, batch, pred_idx, pil_image, img_size=IMG_SIZE):
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=batch, targets=[ClassifierOutputTarget(pred_idx)])[0]
    rgb_img = np.array(pil_image.convert("RGB").resize((img_size, img_size))) / 255.0
    return show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
