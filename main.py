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
zero_shot = None

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
    global classifier, zero_shot
    print("🔄 Loading pre-trained model...")
    print("⏳ This may take 1-2 minutes on first run (downloading model)...")
    
    # Use a lighter model that works with 512MB RAM (MobileNet)
    # This model is smaller (~14MB) and more memory efficient
    classifier = pipeline("image-classification", model="google/mobilenet_v2_1.0_224")
    # Zero-shot image classification for vegetable identification and freshness
    zero_shot = pipeline(
        "zero-shot-image-classification",
        model="openai/clip-vit-base-patch32"
    )
    
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
            "general_classifier": "google/mobilenet_v2_1.0_224",
            "zero_shot": "openai/clip-vit-base-patch32"
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
        "model_loaded": classifier is not None,
        "zero_shot_loaded": zero_shot is not None
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


@app.post("/detect")
async def detect_vegetable(file: UploadFile = File(...)):
    """
    Detect vegetable name and freshness state using zero-shot image classification.
    Returns top vegetable and freshness (fresh or damaged) with confidences.
    """
    if zero_shot is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please wait...")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Identify vegetable type
        veg_candidates = [f"a photo of {lbl}" for lbl in VEGETABLE_LABELS]
        veg_result = zero_shot(
            image,
            candidate_labels=veg_candidates,
            hypothesis_template="{}"
        )

        # zero-shot returns list sorted by score
        top_label_raw = veg_result[0]["label"]
        top_score = float(veg_result[0]["score"])
        # Normalize label back (strip template)
        vegetable = top_label_raw.replace("a photo of ", "").strip()

        # Freshness estimation
        freshness_candidates = [
            f"a photo of fresh {vegetable}",
            f"a photo of damaged {vegetable}",
            f"a photo of rotten {vegetable}",
            f"a photo of bruised {vegetable}"
        ]
        freshness_result = zero_shot(
            image,
            candidate_labels=freshness_candidates,
            hypothesis_template="{}"
        )
        # Aggregate damaged-related labels
        fresh_score = 0.0
        damaged_score = 0.0
        for item in freshness_result:
            lbl = item["label"].lower()
            if "fresh" in lbl:
                fresh_score = max(fresh_score, float(item["score"]))
            else:
                damaged_score = max(damaged_score, float(item["score"]))

        freshness_status = "fresh" if fresh_score >= damaged_score else "damaged"
        freshness_confidence = round(max(fresh_score, damaged_score) * 100, 2)

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
