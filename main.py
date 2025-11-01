"""
Image Classification API using Pre-trained MobileNet Model
Model: google/mobilenet_v2_1.0_224 (optimized for low memory)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from transformers import pipeline
from PIL import Image
import io
import uvicorn
from pathlib import Path
from typing import List
import numpy as np
import torch

# Initialize FastAPI app
app = FastAPI(
    title="Vegetable Detection API",
    description="AI-powered vegetable identification and freshness estimation",
    version="1.0.0"
)

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
classifier = None

# Common vegetable labels (curated, compact list for zero-shot classification)
VEGETABLE_LABELS: List[str] = [
    "tomato", "cucumber", "zucchini", "eggplant", "aubergine", "carrot", "potato",
    "sweet potato", "onion", "garlic", "bell pepper", "green pepper", "red pepper",
    "chili pepper", "broccoli", "cauliflower", "cabbage", "lettuce", "spinach",
    "kale", "bok choy", "okra", "pumpkin", "butternut squash", "peas", "green beans",
    "asparagus", "radish", "beet", "beetroot", "ginger", "corn", "maize"
]

@app.on_event("startup")
async def load_model():
    """Load the pre-trained model on startup"""
    global classifier
    print("🔄 Loading pre-trained model...")
    print("⏳ This may take 1-2 minutes on first run (downloading model)...")
    
    # Reduce CPU thread usage for lower memory/CPU footprint
    try:
        torch.set_grad_enabled(False)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    # Use a lighter model that works with low memory (MobileNet)
    # This model is smaller (~14MB) and efficient
    classifier = pipeline("image-classification", model="google/mobilenet_v2_1.0_224")
    
    print("✅ Model loaded successfully!")
    print("🚀 API is ready to classify images!")


@app.get("/")
@app.head("/")
async def root():
    """Serve the custom UI"""
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    else:
        # Fallback to API info if HTML not found
        return {
            "message": "Image Classification API",
            "model": "google/mobilenet_v2_1.0_224",
            "status": "running",
            "endpoints": {
                "/classify": "POST - Upload an image to classify",
                "/health": "GET - Check API health status",
                "/docs": "GET - Interactive API documentation"
            }
        }


@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "message": "Vegetable Detection API",
        "models": {
            "general_classifier": "google/mobilenet_v2_1.0_224"
        },
        "status": "running",
        "endpoints": {
            "/classify": "POST - Upload an image to classify",
            "/detect": "POST - Upload an image to detect vegetable and freshness",
            "/health": "GET - Check API health status",
            "/docs": "GET - Interactive API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": classifier is not None
    }


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """
    Classify an uploaded image
    
    Args:
        file: Image file (JPG, PNG, etc.)
    
    Returns:
        JSON with top predictions and confidence scores
    """
    
    # Check if model is loaded
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please wait...")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image file
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary (handles PNG with alpha channel, etc.)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Classify the image (returns top 5 predictions by default)
        predictions = classifier(image)
        
        # Format response
        return {
            "success": True,
            "filename": file.filename,
            "predictions": [
                {
                    "label": pred["label"],
                    "confidence": round(pred["score"] * 100, 2),  # Convert to percentage
                    "score": round(pred["score"], 4)
                }
                for pred in predictions
            ],
            "top_prediction": {
                "label": predictions[0]["label"],
                "confidence": round(predictions[0]["score"] * 100, 2)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


def _analyze_freshness(image: Image.Image, vegetable_hint: str) -> dict:
    """Heuristic freshness estimator using HSV stats (lightweight)."""
    img = image.resize((256, 256))
    hsv = img.convert("HSV")
    arr = np.array(hsv, dtype=np.uint8)
    h = arr[:, :, 0].astype(np.int32)
    s = arr[:, :, 1].astype(np.int32)
    v = arr[:, :, 2].astype(np.int32)

    total = arr.shape[0] * arr.shape[1]
    # Masks (HSV ranges are approximate; PIL H is 0..255)
    dark = (v < 60)
    brown = ((h >= 10) & (h <= 35) & (s >= 40) & (v <= 180))
    green = ((h >= 60) & (h <= 110) & (s >= 50) & (v >= 80))

    dark_ratio = float(dark.sum()) / total
    brown_ratio = float(brown.sum()) / total
    green_ratio = float(green.sum()) / total
    s_mean = float(s.mean()) / 255.0

    # Freshness score: favor vibrant saturation and expected color (green for leafy veggies)
    fresh_base = 0.4 * s_mean + 0.4 * green_ratio
    # Penalties for dark/brown spots
    damage = max(dark_ratio, brown_ratio)
    freshness_score = np.clip(fresh_base - 0.6 * damage + 0.2, 0.0, 1.0)

    status = "fresh" if freshness_score >= 0.5 else "damaged"
    confidence = round(abs(freshness_score - 0.5) * 200, 2)  # distance from 0.5 mapped to %
    return {"status": status, "confidence": confidence}


@app.post("/detect")
async def detect_vegetable(file: UploadFile = File(...)):
    """
    Detect vegetable name and freshness state using zero-shot image classification.
    Returns top vegetable and freshness (fresh or damaged) with confidences.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Use MobileNet predictions, then map to known vegetables
        preds = classifier(image, top_k=5)
        # Find best match among VEGETABLE_LABELS using substring matching
        label_conf = []
        for p in preds:
            lbl = p["label"].lower()
            score = float(p["score"])
            for veg in VEGETABLE_LABELS:
                if veg in lbl:
                    label_conf.append((veg, score))
        if label_conf:
            vegetable, top_score = max(label_conf, key=lambda x: x[1])
        else:
            # fallback to top raw label
            vegetable = preds[0]["label"].split(",")[0].lower()
            top_score = float(preds[0]["score"])

        # Freshness estimation via lightweight heuristics
        fres = _analyze_freshness(image, vegetable)
        freshness_status = fres["status"]
        freshness_confidence = fres["confidence"]

        return {
            "success": True,
            "filename": file.filename,
            "vegetable": vegetable,
            "vegetable_confidence": round(top_score * 100, 2),
            "freshness": {
                "status": freshness_status,
                "confidence": freshness_confidence
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/classify-top")
async def classify_image_top_only(file: UploadFile = File(...)):
    """
    Classify an uploaded image and return only the top prediction
    
    Args:
        file: Image file (JPG, PNG, etc.)
    
    Returns:
        JSON with only the top prediction
    """
    
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please wait...")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        predictions = classifier(image, top_k=1)  # Get only top prediction
        
        return {
            "success": True,
            "filename": file.filename,
            "label": predictions[0]["label"],
            "confidence": round(predictions[0]["score"] * 100, 2),
            "score": round(predictions[0]["score"], 4)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    print("=" * 60)
    print("🤖 Image Classification API")
    print("=" * 60)
    print("📦 Model: google/mobilenet_v2_1.0_224")
    print("🌐 Custom UI: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🔧 API Info: http://localhost:8000/api")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
