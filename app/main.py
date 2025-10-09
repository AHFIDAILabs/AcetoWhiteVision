import sys
from pathlib import Path
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil

# --- Fix for Python Path ---
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
# ---------------------------

from Acetowhite_Vision.pipeline.predict import PredictionPipeline
from Acetowhite_Vision.utils.logger import logger

# Initialize FastAPI
app = FastAPI(title="Acetowhite Vision AI")

# Add CORS middleware (helpful for debugging)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Prediction dashboard."""
    return templates.TemplateResponse("prediction.html", {"request": request})

@app.get("/faq", response_class=HTMLResponse)
async def faq(request: Request):
    """FAQ page."""
    return templates.TemplateResponse("faq.html", {"request": request})

# --- Prediction Endpoint ---

@app.post("/predict")
async def predict_route(file: UploadFile = File(...)): # <-- RENAMED from image_file to file
    file_path = None
    try:
        # Validate file
        if not file or not file.filename:
            return JSONResponse(status_code=400, content={"error": "No file uploaded"})
        if not file.content_type.startswith("image/"):
            return JSONResponse(status_code=415, content={"error": "Invalid file type"})

        # Save temp file
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as f:
            f.write(await file.read())

        logger.info(f"File uploaded successfully: {file_path}")

        # Run prediction
        pipeline = PredictionPipeline(filename=file_path)
        results = pipeline.predict_with_explanation()

        # Clean up
        os.remove(file_path)

        return JSONResponse(content=results)

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- Health Check Endpoint ---

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the API is running."""
    return JSONResponse(content={
        "status": "healthy",
        "message": "Acetowhite Vision AI is running"
    })