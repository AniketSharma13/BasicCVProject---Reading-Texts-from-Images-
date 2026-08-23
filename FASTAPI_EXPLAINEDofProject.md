# Beginner's Visual Guide: How `main.py` Works & Connects FastAPI with AI

Welcome! This guide explains how [`main.py`](file:///Users/aniketsharma/Desktop/PrimeBatchNotes%20/Project%20/CV%20Project/main.py) works from scratch using simple analogies, visual diagrams, and code breakdowns.

---

## 1. The Big Picture 🗺️

Think of your project like a **Restaurant**:
* **The Customer (Frontend / Browser)**: You, uploading an image and wanting to see extracted text.
* **The Waiter (FastAPI in `main.py`)**: Listens for requests, takes your image to the kitchen, and brings back results.
* **The Master Chef (Computer Vision Engine in `backend/`)**: Receives the image, enhances it, detects characters using EasyOCR, and draws bounding boxes.

### Visual Data Flow Diagram:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Your Browser)                         │
│                                                                        │
│   [ Upload Image ] ───▶ [ Click "Extract & Visualize Text" Button ]    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                         HTTP POST Request (FormData)
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         BACKEND SERVER (main.py)                       │
│                                                                        │
│  1. Receives image file via FastAPI endpoint (/api/extract)            │
│  2. Converts uploaded file bytes into OpenCV image matrix (np.ndarray) │
│  3. Passes image matrix to backend.ocr_engine.extract_text()           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        AI OCR MODEL (EasyOCR + OpenCV)                 │
│                                                                        │
│  1. Preprocessing: Converts to Grayscale + Contrast CLAHE + Denoise    │
│  2. EasyOCR Model: Scans image and outputs bounding box coordinates    │
│  3. OpenCV Utility: Draws green boxes & cyan text on original image    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                        Returns JSON Text & PNG Stream
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        FRONTEND DISPLAY (Browser)                      │
│                                                                        │
│  • Extracted Text displayed in text box (with Copy button)             │
│  • Bounding box image rendered in image preview box                    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Concepts in `main.py`

### A. What is FastAPI?
FastAPI is a modern Python framework used to create Web APIs (Application Programming Interfaces). It takes function returns in Python and automatically converts them to web formats (like JSON or image streams).

### B. What is an Endpoint?
An **Endpoint** is a specific URL route where clients send data. In `main.py`, we defined 3 main endpoints:

| Route | Method | Purpose |
| :--- | :--- | :--- |
| **`GET /`** | `GET` | Returns the HTML Web Dashboard page to the browser. |
| **`POST /api/extract`** | `POST` | Receives an uploaded image and returns extracted text as **JSON**. |
| **`POST /api/visualize`** | `POST` | Receives an uploaded image and returns the **annotated PNG image**. |

---

## 3. Step-by-Step Code Breakdown of `main.py`

### Part 1: Imports & Initialization

```python
from fastapi import FastAPI, UploadFile, File, Form, Response
import cv2
import numpy as np
from backend.ocr_engine import extract_text
from backend.utils import draw_detections

# Create the FastAPI app instance
app = FastAPI(title="Computer Vision OCR API")
```
- `app = FastAPI(...)`: Creates our web server application.
- `from backend.ocr_engine import extract_text`: Imports our EasyOCR model engine.
- `from backend.utils import draw_detections`: Imports our bounding box drawing utility.

---

### Part 2: Helper Function to Decode Image Bytes

When a user uploads an image over the internet, it arrives as **raw binary bytes** (`010101...`). OpenCV cannot directly read raw network bytes — it needs an image matrix.

```python
def read_image_bytes(file_bytes: bytes) -> np.ndarray:
    # Convert binary bytes to uint8 numpy array
    nparr = np.frombuffer(file_bytes, np.uint8)
    # Decode array into OpenCV BGR image matrix
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image
```

---

### Part 3: The Text Extraction Endpoint (`POST /api/extract`)

```python
@app.post("/api/extract")
async def api_extract_text(
    file: UploadFile = File(...),              # Uploaded image file
    confidence_threshold: float = Form(0.35),  # Slider value from UI
    use_preprocessing: bool = Form(True)       # Toggle switch from UI
):
    # 1. Read binary bytes from upload
    contents = await file.read()
    
    # 2. Convert bytes to OpenCV image
    image = read_image_bytes(contents)
    
    # 3. Call AI engine
    result = extract_text(image, confidence_threshold=confidence_threshold, use_preprocessing=use_preprocessing)
    
    # 4. Return JSON response
    return {
        "success": True,
        "filename": file.filename,
        "extracted_text": result["text"],
        "count": result["count"],
        "detections": result["detections"]
    }
```

---

### Part 4: The Image Visualization Endpoint (`POST /api/visualize`)

```python
@app.post("/api/visualize")
async def api_visualize(file: UploadFile = File(...), ...):
    contents = await file.read()
    image = read_image_bytes(contents)
    
    # 1. Get OCR detections
    result = extract_text(image, ...)
    
    # 2. Draw green bounding boxes on the image
    annotated_image = draw_detections(image, result["detections"])

    # 3. Compress OpenCV image matrix back into PNG bytes
    is_success, buffer = cv2.imencode(".png", annotated_image)

    # 4. Return PNG binary stream directly to the browser
    return Response(content=buffer.tobytes(), media_type="image/png")
```

---

## 4. How the HTML/JavaScript UI Connects to FastAPI 🔗

In `main.py`, the `GET /` endpoint returns an embedded HTML page containing JavaScript. Here is how the browser communicates with FastAPI:

### 1. Preparing the Upload Form Data in JavaScript:

```javascript
// Collect user input from UI controls
const formData = new FormData();
formData.append('file', selectedFile);                          // Uploaded image
formData.append('confidence_threshold', thresholdSlider.value); // e.g., 0.35
formData.append('use_preprocessing', preprocessCheckbox.checked); // true/false
```

### 2. Request 1: Fetching Text (JSON)

```javascript
const response = await fetch('/api/extract', {
    method: 'POST',
    body: formData
});

const data = await response.json();
// Update text box on UI
document.getElementById('outputContainer').innerText = data.extracted_text;
```

### 3. Request 2: Fetching Visual Image (PNG Blob)

```javascript
const vizResponse = await fetch('/api/visualize', {
    method: 'POST',
    body: formData
});

// Convert image response stream to browser URL
const imageBlob = await vizResponse.blob();
const objectUrl = URL.createObjectURL(imageBlob);

// Display image in <img> tag
document.getElementById('resultImage').src = objectUrl;
```

---

## Summary Checklist for Beginners 💡

1. **`main.py` is the bridge**: It connects your user interface (HTML/JS) to your computer vision algorithms (`backend/`).
2. **`FastAPI` handles HTTP routes**: Endpoints listen for requests and parse inputs like files and form sliders.
3. **`OpenCV (cv2)` handles image pixels**: `cv2.imdecode` reads uploaded files into matrices, and `cv2.imencode` packages them back to PNG files.
4. **`fetch()` in JavaScript is the telephone**: It sends requests from the webpage to FastAPI without reloading the browser tab!
