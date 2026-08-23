import cv2

def draw_detections(image, detections):
    """
    Draws bounding boxes and text labels on image.
    If detections list is empty, overlays 'No text found!' on the image.
    """
    annotated_image = image.copy()

    if not detections:
        h, w, _ = annotated_image.shape
        cv2.putText(
            annotated_image,
            "No text found!",
            (int(w * 0.1), int(h * 0.5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )
        return annotated_image

    for detection in detections:
        bbox = detection["bbox"]
        points = [tuple(point) for point in bbox]

        # Draw bounding box rectangle (polygon using 4 corner points)
        for i in range(4):
            cv2.line(
                annotated_image,
                points[i],
                points[(i + 1) % 4],
                (0, 255, 0),
                2
            )

        text = detection["text"]
        x, y = points[0]

        # Position label slightly above top-left corner
        cv2.putText(
            annotated_image,
            text,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

    return annotated_image