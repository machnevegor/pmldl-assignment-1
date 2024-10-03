import io
import logging
import os
from enum import StrEnum

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

# --- LOGGER --- #

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- DEVICE --- #


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"  # Apple GPU
    else:
        return "cpu"


device = get_device()
logger.info(f"Using device: {device}")

# --- CONFIG --- #

CONFIG = {
    "ckpt_path": os.getenv("CKPT_PATH"),
    "device": device,
}


# --- MODEL --- #


class CNNClassificationModel(nn.Module):
    def __init__(self, num_classes=1):
        super(CNNClassificationModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                3, 32, kernel_size=3, stride=1, padding=1
            ),  # [3, 64, 64] -> [32, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [32, 64, 64] -> [32, 32, 32]
            nn.Conv2d(
                32, 64, kernel_size=3, stride=1, padding=1
            ),  # [32, 32, 32] -> [64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [64, 32, 32] -> [64, 16, 16]
            nn.Conv2d(
                64, 128, kernel_size=3, stride=1, padding=1
            ),  # [64, 16, 16] -> [128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [128, 16, 16] -> [128, 8, 8]
            nn.Conv2d(
                128, 256, kernel_size=3, stride=1, padding=1
            ),  # [128, 8, 8] -> [256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [256, 8, 8] -> [256, 4, 4]
            nn.Flatten(),  # [256, 4, 4] -> [4096]
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.model(x)


def load_model(model_path: str, device: str) -> nn.Module:
    """Load the pre-trained model from the specified path."""
    model = CNNClassificationModel().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise RuntimeError(f"Failed to load model from {model_path}")

    return model


data_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ]
)


async def preprocess_image(file: UploadFile) -> torch.Tensor:
    """Read and preprocess the uploaded image."""
    try:
        image = await file.read()
        img = Image.open(io.BytesIO(image)).convert("RGB")
        img_tensor = data_transforms(img).unsqueeze(0).to(device)
        return img_tensor
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image data")


class HairColor(StrEnum):
    BLONDE = "Blonde"
    NOT_BLONDE = "Not blonde"


async def predict_image(model: nn.Module, img_tensor: torch.Tensor) -> str:
    """Perform inference on the image tensor and return the prediction."""
    with torch.no_grad():
        output = model(img_tensor)
        prediction_value = (output > 0.5).float().item()
        return HairColor.BLONDE if prediction_value else HairColor.NOT_BLONDE


# --- SERVER --- #

app = FastAPI()

model = load_model(CONFIG["ckpt_path"], CONFIG["device"])


class PredictResponse(BaseModel):
    prediction: HairColor


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile) -> PredictResponse:
    """Handle prediction requests with an uploaded image."""
    img_tensor = await preprocess_image(file)

    prediction = await predict_image(model, img_tensor)

    return PredictResponse(prediction=prediction)
