import cv2
import matplotlib.pyplot as plt
import easyocr
import numpy as np
import os

def preprocess_image(image):
    """
    Preprocesses input image to optimize OCR text extraction:
    1. Converts to Grayscale.
    2. Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast.
    3. Applies Bilateral Filtering to remove noise while keeping text edges sharp.
    """
    # 1. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Adaptive Contrast Enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)
    
    # 3. Bilateral Noise Reduction (preserves text edges)
    denoised = cv2.bilateralFilter(contrast_enhanced, 9, 75, 75)
    
    return denoised
