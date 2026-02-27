"""
backend.py — FastAPI backend server for Tamil Handwritten Text Extraction.

Provides REST API endpoints for:
  - POST /api/extract     → Extract text from an uploaded image
  - GET  /api/models      → List available models
  - GET  /api/health      → Health check

Run with:
    uvicorn backend:app --reload --port 8000
"""

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
import os
import tempfile
import time

# ── FastAPI App ───────────────────────────────────────────────────────
app = FastAPI(
    title="Tamil Handwritten Text Extractor API",
    description="AI-powered extraction of handwritten Tamil text from images using EasyOCR + CNN + MobileNetV2",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS — allow Streamlit frontend and any local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model cache ────────────────────────────────────────────────
_extractor_cache = {}


# ── Response Models ───────────────────────────────────────────────────
class TextRegion(BaseModel):
    label: str
    confidence: float
    bbox: list  # [x, y, w, h]


class ExtractionResult(BaseModel):
    text: str
    regions: list[TextRegion]
    num_regions: int
    avg_confidence: float
    model_used: str
    processing_time_ms: float


class ModelInfo(BaseModel):
    name: str
    type: str
    status: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: int


# ── Helper ────────────────────────────────────────────────────────────
def get_extractor(model_type="easyocr"):
    """Get or create a cached extractor instance."""
    if model_type not in _extractor_cache:
        from src.extractor import get_available_models, load_extractor
        models = get_available_models()

        # Find matching model
        for name, info in models.items():
            if info['type'] == model_type:
                _extractor_cache[model_type] = load_extractor(info)
                break
        else:
            # Default to easyocr
            for name, info in models.items():
                if info['type'] == 'easyocr':
                    _extractor_cache[model_type] = load_extractor(info)
                    break

    return _extractor_cache.get(model_type)


# ── API Endpoints ─────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check if the API server is running and models are loaded."""
    return HealthResponse(
        status="ok",
        models_loaded=len(_extractor_cache)
    )


@app.get("/api/models", response_model=list[ModelInfo], tags=["Models"])
async def list_models():
    """List all available OCR models and their status."""
    from src.extractor import get_available_models
    models = get_available_models()
    result = []
    for name, info in models.items():
        status = "ready" if info['type'] == 'easyocr' else "requires_training"
        if info.get('path') and os.path.exists(info['path']):
            status = "ready"
        result.append(ModelInfo(name=name, type=info['type'], status=status))
    return result


@app.post("/api/extract", response_model=ExtractionResult, tags=["Extraction"])
async def extract_text(
    file: UploadFile = File(..., description="Image file containing handwritten Tamil text"),
    model: str = Query("easyocr", description="Model to use: easyocr, cnn, or mobilenet"),
    confidence_threshold: float = Query(0.3, ge=0.0, le=1.0, description="Minimum confidence threshold")
):
    """
    Extract Tamil text from an uploaded handwritten image.

    - **file**: Upload a JPG, PNG, or BMP image
    - **model**: Choose 'easyocr' (default), 'cnn', or 'mobilenet'
    - **confidence_threshold**: Filter results below this confidence (0.0 - 1.0)
    """
    start_time = time.time()

    # Read uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid image file. Please upload a JPG, PNG, or BMP image."}
        )

    # Save to temp file (EasyOCR needs a file path)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        cv2.imwrite(tmp.name, img)
        temp_path = tmp.name

    try:
        # Get extractor and extract text
        extractor = get_extractor(model)
        if extractor is None:
            return JSONResponse(
                status_code=400,
                content={"error": f"Model '{model}' is not available. Use 'easyocr'."}
            )

        text, details = extractor.predict_text(temp_path, confidence_threshold=confidence_threshold)

        # Build response
        regions = [
            TextRegion(
                label=d['label'],
                confidence=d['confidence'],
                bbox=list(d['bbox'])
            )
            for d in details
        ]

        avg_conf = np.mean([d['confidence'] for d in details]) if details else 0.0
        processing_time = (time.time() - start_time) * 1000

        return ExtractionResult(
            text=text,
            regions=regions,
            num_regions=len(regions),
            avg_confidence=round(float(avg_conf), 4),
            model_used=model,
            processing_time_ms=round(processing_time, 2)
        )

    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ── Run ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI backend server...")
    print("API docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
