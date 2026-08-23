import os
import sys
import io
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Response, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Ensure project root is in sys.path for clean imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.ocr_engine import extract_text
from backend.utils import draw_detections

app = FastAPI(
    title="Computer Vision OCR API",
    description="FastAPI interface for image text detection and extraction using EasyOCR & OpenCV preprocessing.",
    version="1.0.0"
)

# Enable CORS for external client applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def read_image_bytes(file_bytes: bytes) -> np.ndarray:
    """Utility function to decode image bytes into an OpenCV BGR matrix."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file format or corrupted bytes.")
    return image

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint to verify backend service status."""
    return {
        "status": "online",
        "service": "EasyOCR Computer Vision API",
        "version": "1.0.0"
    }

@app.post("/api/extract", tags=["OCR Engine"])
async def api_extract_text(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.35),
    use_preprocessing: bool = Form(True)
):
    """
    Extracts text from uploaded image file.
    Returns JSON response containing extracted text, detection count, and detailed bounding boxes.
    """
    try:
        contents = await file.read()
        image = read_image_bytes(contents)
        result = extract_text(image, confidence_threshold=confidence_threshold, use_preprocessing=use_preprocessing)
        return {
            "success": True,
            "filename": file.filename,
            "extracted_text": result["text"],
            "count": result["count"],
            "detections": result["detections"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/visualize", tags=["OCR Engine"])
async def api_visualize(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.35),
    use_preprocessing: bool = Form(True)
):
    """
    Extracts text and renders bounding box annotations on the image.
    Returns the annotated PNG image stream directly.
    """
    try:
        contents = await file.read()
        image = read_image_bytes(contents)
        result = extract_text(image, confidence_threshold=confidence_threshold, use_preprocessing=use_preprocessing)
        annotated_image = draw_detections(image, result["detections"])

        is_success, buffer = cv2.imencode(".png", annotated_image)
        if not is_success:
            raise HTTPException(status_code=500, detail="Failed to encode annotated image.")

        return Response(content=buffer.tobytes(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse, tags=["Web UI"])
async def serve_ui():
    """Serves an interactive Web UI for testing and running the OCR model."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Optical Character Recognition (OCR)</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
                --card-bg: rgba(30, 41, 59, 0.7);
                --card-border: rgba(255, 255, 255, 0.1);
                --accent-color: #6366f1;
                --accent-hover: #4f46e5;
                --text-primary: #f8fafc;
                --text-secondary: #94a3b8;
                --success-color: #10b981;
            }

            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
                font-family: 'Inter', sans-serif;
            }

            body {
                background: var(--bg-gradient);
                color: var(--text-primary);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 2rem 1rem;
            }

            .header {
                text-align: center;
                margin-bottom: 2rem;
            }

            .header h1 {
                font-size: 2.25rem;
                font-weight: 700;
                background: linear-gradient(to right, #818cf8, #c084fc);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.5rem;
            }

            .header p {
                color: var(--text-secondary);
                font-size: 1rem;
            }

            .container {
                width: 100%;
                max-width: 1100px;
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1.5rem;
            }

            @media (max-width: 768px) {
                .container {
                    grid-template-columns: 1fr;
                }
            }

            .card {
                background: var(--card-bg);
                backdrop-filter: blur(12px);
                border: 1px solid var(--card-border);
                border-radius: 16px;
                padding: 1.5rem;
                display: flex;
                flex-direction: column;
            }

            .card-title {
                font-size: 1.1rem;
                font-weight: 600;
                margin-bottom: 1rem;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }

            .upload-area {
                border: 2px dashed rgba(99, 102, 241, 0.4);
                border-radius: 12px;
                padding: 2.5rem 1.5rem;
                text-align: center;
                cursor: pointer;
                transition: all 0.2s ease;
                background: rgba(15, 23, 42, 0.4);
            }

            .upload-area:hover, .upload-area.dragover {
                border-color: var(--accent-color);
                background: rgba(99, 102, 241, 0.1);
            }

            .upload-icon {
                font-size: 2.5rem;
                margin-bottom: 0.75rem;
                display: block;
            }

            .controls {
                margin-top: 1rem;
                display: flex;
                flex-direction: column;
                gap: 0.75rem;
            }

            .control-group {
                display: flex;
                align-items: center;
                justify-content: space-between;
                font-size: 0.875rem;
                color: var(--text-secondary);
            }

            input[type="range"] {
                accent-color: var(--accent-color);
            }

            .btn {
                background: var(--accent-color);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0.75rem 1rem;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: background 0.2s ease;
                width: 100%;
                margin-top: 0.5rem;
            }

            .btn:hover {
                background: var(--accent-hover);
            }

            .btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }

            .text-box {
                background: rgba(15, 23, 42, 0.6);
                border: 1px solid var(--card-border);
                border-radius: 8px;
                padding: 1rem;
                font-family: monospace;
                font-size: 0.95rem;
                color: #e2e8f0;
                min-height: 180px;
                max-height: 250px;
                overflow-y: auto;
                white-space: pre-wrap;
                margin-bottom: 1rem;
            }

            .copy-btn {
                background: rgba(255, 255, 255, 0.1);
                color: var(--text-primary);
                border: 1px solid var(--card-border);
                padding: 0.35rem 0.75rem;
                border-radius: 6px;
                font-size: 0.8rem;
                cursor: pointer;
            }

            .copy-btn:hover {
                background: rgba(255, 255, 255, 0.2);
            }

            .preview-img {
                width: 100%;
                max-height: 350px;
                object-fit: contain;
                border-radius: 8px;
                border: 1px solid var(--card-border);
                background: rgba(0, 0, 0, 0.2);
            }

            .spinner {
                display: inline-block;
                width: 1.25rem;
                height: 1.25rem;
                border: 3px solid rgba(255,255,255,0.3);
                border-radius: 50%;
                border-top-color: white;
                animation: spin 0.8s linear infinite;
            }

            @keyframes spin {
                to { transform: rotate(360deg); }
            }

            .badge {
                background: rgba(99, 102, 241, 0.2);
                color: #a5b4fc;
                font-size: 0.75rem;
                padding: 0.2rem 0.5rem;
                border-radius: 4px;
            }

            .disclaimer-banner {
                background: rgba(234, 179, 8, 0.12);
                border: 1px solid rgba(234, 179, 8, 0.3);
                color: #fef08a;
                font-size: 0.8rem;
                padding: 0.6rem 0.75rem;
                border-radius: 8px;
                margin-top: 0.75rem;
                text-align: center;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 0.4rem;
            }

            .disclaimer-banner code {
                background: rgba(0, 0, 0, 0.3);
                padding: 0.1rem 0.3rem;
                border-radius: 4px;
                font-family: monospace;
            }
        </style>
    </head>
    <body>

        <div class="header">
            <h1>OCR Vision Dashboard</h1>
            <p>Upload any image to extract text and visualize computer vision bounding boxes</p>
        </div>

        <div class="container">
            <!-- Left Panel: Input & Control -->
            <div class="card">
                <div class="card-title">
                    <span>Source Image</span>
                    <span class="badge" id="fileStatus">Ready</span>
                </div>

                <div class="upload-area" id="dropZone" onclick="document.getElementById('fileInput').click()">
                    <span class="upload-icon">📷</span>
                    <p style="font-weight: 500;">Click to upload or drag & drop</p>
                    <p style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.25rem;">Supported formats: .jpg, .jpeg, .png, .bmp, .webp</p>
                    <input type="file" id="fileInput" accept=".jpg,.jpeg,.png,.bmp,.webp,image/jpeg,image/png,image/bmp,image/webp" style="display: none;" onchange="handleFileSelect(event)">
                </div>

                <div class="disclaimer-banner">
                    ⚠️ <span><strong>Format Notice:</strong> Only <code>.jpg</code>, <code>.jpeg</code>, <code>.png</code>, <code>.bmp</code>, and <code>.webp</code> formats are supported.</span>
                </div>

                <div class="controls">
                    <div class="control-group">
                        <label for="threshold">Confidence Threshold: <span id="threshVal">0.35</span></label>
                        <input type="range" id="threshold" min="0.1" max="0.9" step="0.05" value="0.35" oninput="document.getElementById('threshVal').innerText = this.value">
                    </div>

                    <div class="control-group">
                        <label for="preprocess">Image Preprocessing (CLAHE + Denoise)</label>
                        <input type="checkbox" id="preprocess" checked style="accent-color: var(--accent-color);">
                    </div>

                    <button class="btn" id="submitBtn" onclick="runOCR()">Extract & Visualize Text</button>
                </div>
            </div>

            <!-- Right Panel: Output & Visualization -->
            <div class="card">
                <div class="card-title">
                    <span>Extracted Output</span>
                    <button class="copy-btn" onclick="copyText()">Copy Text</button>
                </div>

                <div class="text-box" id="outputContainer">Select an image and click 'Extract & Visualize Text'</div>

                <div class="card-title" style="margin-top: 0.5rem;">
                    <span>Annotated Visual</span>
                </div>

                <img id="resultImage" class="preview-img" style="display: none;" alt="OCR Annotated Result">
                <div id="imagePlaceholder" style="text-align: center; color: var(--text-secondary); padding: 3rem 0; border: 1px dashed var(--card-border); border-radius: 8px;">
                    Annotated bounding box image will appear here
                </div>
            </div>
        </div>

        <script>
            let selectedFile = null;
            const ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'bmp', 'webp'];

            function validateAndSetFile(file) {
                if (!file) return false;
                const ext = file.name.split('.').pop().toLowerCase();
                if (!ALLOWED_EXTENSIONS.includes(ext)) {
                    alert(`Unsupported file format (.${ext})!\n\nPlease upload an image file in .jpg, .jpeg, .png, .bmp, or .webp format.`);
                    return false;
                }
                selectedFile = file;
                document.getElementById('fileStatus').innerText = file.name;
                document.getElementById('dropZone').style.borderColor = '#10b981';
                return true;
            }

            function handleFileSelect(e) {
                const file = e.target.files[0];
                validateAndSetFile(file);
            }

            const dropZone = document.getElementById('dropZone');
            ['dragenter', 'dragover'].forEach(eventName => {
                dropZone.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    dropZone.classList.add('dragover');
                }, false);
            });

            ['dragleave', 'drop'].forEach(eventName => {
                dropZone.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    dropZone.classList.remove('dragover');
                }, false);
            });

            dropZone.addEventListener('drop', (e) => {
                const dt = e.dataTransfer;
                const files = dt.files;
                if (files.length > 0) {
                    validateAndSetFile(files[0]);
                }
            });

            async function runOCR() {
                if (!selectedFile) {
                    alert('Please select or drop an image file first!');
                    return;
                }

                const btn = document.getElementById('submitBtn');
                const output = document.getElementById('outputContainer');
                const resultImg = document.getElementById('resultImage');
                const placeholder = document.getElementById('imagePlaceholder');

                btn.disabled = true;
                btn.innerHTML = '<span class="spinner"></span> Processing OCR...';
                output.innerText = 'Extracting text with OpenCV & EasyOCR...';

                const thresh = document.getElementById('threshold').value;
                const preprocess = document.getElementById('preprocess').checked;

                const formData = new FormData();
                formData.append('file', selectedFile);
                formData.append('confidence_threshold', thresh);
                formData.append('use_preprocessing', preprocess);

                try {
                    // 1. Fetch text extraction JSON
                    const response = await fetch('/api/extract', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();

                    if (data.count === 0) {
                        output.innerText = "No text found!";
                    } else {
                        output.innerText = data.extracted_text;
                    }

                    // 2. Fetch annotated visualization image
                    const vizResponse = await fetch('/api/visualize', {
                        method: 'POST',
                        body: formData
                    });
                    const imageBlob = await vizResponse.blob();
                    const objectUrl = URL.createObjectURL(imageBlob);

                    resultImg.src = objectUrl;
                    resultImg.style.display = 'block';
                    placeholder.style.display = 'none';

                } catch (err) {
                    output.innerText = 'Error running OCR: ' + err.message;
                } finally {
                    btn.disabled = false;
                    btn.innerText = 'Extract & Visualize Text';
                }
            }

            function copyText() {
                const text = document.getElementById('outputContainer').innerText;
                if (text && text !== "Select an image and click 'Extract & Visualize Text'") {
                    navigator.clipboard.writeText(text);
                    alert('Copied to clipboard!');
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    print("Starting FastAPI Computer Vision OCR Server at http://127.0.0.1:8000...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
