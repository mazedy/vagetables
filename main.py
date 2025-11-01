"""
Memory-optimized Image Classification API using MobileNet and Lazy Zero-Shot CLIP
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from PIL import Image
import io
from pathlib import Path
from typing import List

# Initialize FastAPI app
app = FastAPI(
    title="Vegetable Detection API",
    description="AI-powered vegetable identification and freshness estimation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
classifier = None
zero_shot = None

# Compact vegetable labels
VEGETABLE_LABELS: List[str] = [
    "tomato", "cucumber", "zucchini", "eggplant", "aubergine", "carrot", "potato",
    "sweet potato", "onion", "garlic", "bell pepper", "green pepper", "red pepper",
    "chili pepper", "broccoli", "cauliflower", "cabbage", "lettuce", "spinach",
    "kale", "bok choy", "okra", "pumpkin", "butternut squash", "peas", "green beans",
    "asparagus", "radish", "beet", "beetroot", "ginger", "corn", "maize"
]

# --- Startup: Load lightweight MobileNet only ---
@app.on_event("startup")
async def load_model():
    global classifier
    print("🔄 Loading lightweight MobileNet classifier...")
    classifier = pipeline("image-classification", model="google/mobilenet_v2_1.0_224")
    print("✅ MobileNet loaded successfully!")


# --- Lazy load zero-shot model only when first requested ---
def get_zero_shot_pipeline():
    global zero_shot
    if zero_shot is None:
        print("🔄 Loading zero-shot CLIP model (lazy load)...")
        zero_shot = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")
        print("✅ Zero-shot model loaded!")
    return zero_shot


# --- Utility: Resize image to reduce memory usage ---
def prepare_image(image_bytes: bytes, max_size=224):
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.thumbnail((max_size, max_size))
    return image


# --- Health check ---
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "classifier_loaded": classifier is not None,
        "zero_shot_loaded": zero_shot is not None
    }


# --- Image classification endpoint ---
@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier not loaded yet.")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    image = prepare_image(await file.read())
    predictions = classifier(image)
    
    return {
        "success": True,
        "filename": file.filename,
        "predictions": [
            {"label": p["label"], "confidence": round(p["score"] * 100, 2)}
            for p in predictions
        ],
        "top_prediction": {
            "label": predictions[0]["label"],
            "confidence": round(predictions[0]["score"] * 100, 2)
        }
    }


# --- Zero-shot vegetable detection endpoint ---
@app.post("/detect")
async def detect_vegetable(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    zero_shot_pipeline = get_zero_shot_pipeline()
    image = prepare_image(await file.read())
    
    # Vegetable type
    veg_candidates = [f"a photo of {lbl}" for lbl in VEGETABLE_LABELS]
    veg_result = zero_shot_pipeline(image, candidate_labels=veg_candidates, hypothesis_template="{}")
    
    top_label_raw = veg_result[0]["label"]
    top_score = float(veg_result[0]["score"])
    vegetable = top_label_raw.replace("a photo of ", "").strip()
    
    # Freshness estimation
    freshness_candidates = [
        f"a photo of fresh {vegetable}",
        f"a photo of damaged {vegetable}",
        f"a photo of rotten {vegetable}",
        f"a photo of bruised {vegetable}"
    ]
    freshness_result = zero_shot_pipeline(image, candidate_labels=freshness_candidates, hypothesis_template="{}")
    
    fresh_score = max([r["score"] for r in freshness_result if "fresh" in r["label"].lower()] or [0])
    damaged_score = max([r["score"] for r in freshness_result if "fresh" not in r["label"].lower()] or [0])
    
    freshness_status = "fresh" if fresh_score >= damaged_score else "damaged"
    freshness_confidence = round(max(fresh_score, damaged_score) * 100, 2)
    
    return {
        "success": True,
        "filename": file.filename,
        "vegetable": vegetable,
        "vegetable_confidence": round(top_score * 100, 2),
        "freshness": {"status": freshness_status, "confidence": freshness_confidence}
    }


# --- Top prediction only endpoint ---
@app.post("/classify-top")
async def classify_image_top_only(file: UploadFile = File(...)):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier not loaded yet.")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    image = prepare_image(await file.read())
    predictions = classifier(image, top_k=1)
    
    return {
        "success": True,
        "filename": file.filename,
        "label": predictions[0]["label"],
        "confidence": round(predictions[0]["score"] * 100, 2),
        "score": round(predictions[0]["score"], 4)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
