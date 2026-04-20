import os
import io
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
from PIL import Image
from torchvision import transforms

from models.network import SelfPruningNetwork
from utils.logger import setup_logger

logger = setup_logger("fastapi")

# Global variables to hold the model and device
model = None
device = None

# CIFAR-10 class labels
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI to load the model on startup."""
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model on {device}")
    
    model = SelfPruningNetwork().to(device)
    checkpoint_path = "checkpoints/latest_checkpoint.pt"
    
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Apply hard pruning for actual inference speedup/memory saving
        threshold = 0.01  # Assuming 0.01 threshold from our experiments
        prune_results = model.hard_prune_all(threshold=threshold)
        logger.info("Applied hard pruning to loaded model for efficient inference.")
    else:
        logger.warning("No checkpoint found. Using untrained model! Please run training first.")
        
    model.eval()
    yield
    # Cleanup on shutdown
    model = None

app = FastAPI(
    title="Self-Pruning Network API",
    description="API for running inference on a dynamically pruned CIFAR-10 model. Demonstrates real-world edge deployment.",
    version="1.0.0",
    lifespan=lifespan
)

def transform_image(image_bytes):
    """Convert raw bytes to CIFAR-10 normalized tensor."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        return transform(image).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        logger.error(f"Image transformation error: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image data")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image file and returns the predicted CIFAR-10 class.
    Demonstrates low-latency inference using the pruned model.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    image_bytes = await file.read()
    tensor = transform_image(image_bytes).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted_idx = outputs.max(1)
        
    predicted_class = CLASSES[predicted_idx.item()]
    
    # Calculate current compression ratio to show off in the response
    params_info = model.count_parameters()
    
    return {
        "prediction": predicted_class,
        "class_id": predicted_idx.item(),
        "model_efficiency": {
            "total_parameters": params_info["total"],
            "active_parameters": params_info["nonzero"],
            "compression_ratio": round(params_info["compression"], 2)
        }
    }

@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/")
def root():
    """Redirect to the docs page."""
    return RedirectResponse(url="/docs")
