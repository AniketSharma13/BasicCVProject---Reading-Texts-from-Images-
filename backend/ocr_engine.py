import easyocr
from backend.preprocessing import preprocess_image


# Load the OCR model once
reader = easyocr.Reader(['en'],gpu=False)


def extract_text(
    image,
    confidence_threshold=0.35,
    use_preprocessing=True
):
    """
    Extract text from an image.

    Args:
        image: OpenCV image in BGR format.
        confidence_threshold: Minimum OCR confidence.
        use_preprocessing: Whether to preprocess the image.

    Returns:
        Dictionary containing extracted text
        and individual OCR detections.
    """

    if use_preprocessing:
        input_image = preprocess_image(image)
    else:
        input_image = image

    results = reader.readtext(input_image)

    valid_detections = []

    for bbox, text, confidence in results:

        if confidence >= confidence_threshold:

            detection = {
                "text": text,
                "confidence": float(confidence),
                "bbox": [
                    [int(point[0]), int(point[1])]
                    for point in bbox
                ]
            }

            valid_detections.append(detection)

    extracted_text = "\n".join(
        detection["text"]
        for detection in valid_detections
    )

    return {
        "text": extracted_text,
        "detections": valid_detections,
        "count": len(valid_detections)
    }