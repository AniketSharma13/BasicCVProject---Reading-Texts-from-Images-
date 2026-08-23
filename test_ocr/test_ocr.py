import os
import sys
import glob
import cv2
import matplotlib.pyplot as plt

# Ensure project root directory is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.ocr_engine import extract_text
from backend.utils import draw_detections

def run_test_on_image(image_path, output_dir=None, display=True):
    """
    Runs OCR on a single image file, prints results, and visualizes/saves output.
    """
    image_name = os.path.basename(image_path)
    print(f"\n========================================")
    print(f" Processing: {image_name}")
    print(f"========================================")

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from '{image_path}'. Skipping.\n")
        return

    # Run OCR extraction engine
    result = extract_text(image, confidence_threshold=0.35, use_preprocessing=True)

    print("\n[RESULT]")
    print(f"Number of text detections: {result['count']}")

    if result['count'] == 0:
        print("\nNo text found!")
    else:
        print("\nExtracted Text:")
        print(result["text"])

        print("\nIndividual Detections:")
        for idx, detection in enumerate(result["detections"], 1):
            print(f"  {idx}. Text: '{detection['text']}' | Confidence: {detection['confidence']:.2%}")
            print(f"     Bounding Box: {detection['bbox']}")
            print("-" * 40)

    # Draw detections or 'No text found!' overlay
    annotated_image = draw_detections(image, result["detections"])

    # Save output to output_results directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"result_{image_name}")
        cv2.imwrite(save_path, annotated_image)
        print(f"\n[SAVED] Annotated result saved to: {save_path}")

    # Display visualization
    if display:
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        plt.title(f"OCR Result: {image_name} ({result['count']} detections)")
        plt.axis("off")
        plt.show()

def main():
    # Setup test_images and output_results directory paths
    test_images_dir = os.path.join(PROJECT_ROOT, "test_images")
    output_dir = os.path.join(PROJECT_ROOT, "output_results")

    # If specific image path passed via command line argument, process it
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
        if os.path.isfile(target_path):
            run_test_on_image(target_path, output_dir=output_dir, display=True)
            return
        elif os.path.isdir(target_path):
            test_images_dir = target_path

    # Collect all image files in test_images folder
    valid_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    image_paths = []
    for ext in valid_extensions:
        image_paths.extend(glob.glob(os.path.join(test_images_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(test_images_dir, ext.upper())))

    image_paths = sorted(list(set(image_paths)))

    if not image_paths:
        print(f"No test images found in '{test_images_dir}'. Please place images in this directory.")
        return

    print(f"Found {len(image_paths)} image(s) in '{test_images_dir}'. Starting batch OCR testing...\n")

    for img_path in image_paths:
        run_test_on_image(img_path, output_dir=output_dir, display=True)

if __name__ == "__main__":
    main()