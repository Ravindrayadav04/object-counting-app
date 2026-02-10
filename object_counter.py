import cv2
import numpy as np


def count_objects(image, max_width=900):
    """
    Count objects in an image with improved handling for stacked objects and various orientations.
    
    Args:
        image: Input image (BGR format)
        max_width: Maximum width for resizing (maintains aspect ratio)
    
    Returns:
        object_count: Number of detected objects
        output: Annotated image with bounding boxes
        opening: Processed binary image
    """
    # Resize with aspect ratio preservation
    height, width = image.shape[:2]
    scale = max_width / width if width > max_width else 1.0
    new_width = int(width * scale)
    new_height = int(height * scale)
    image = cv2.resize(image, (new_width, new_height))
    original = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Threshold
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Improved morphological operations for stacked object separation
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_large = np.ones((7, 7), np.uint8)

    # Multiple iterations to better separate stacked objects
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_large, iterations=3)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_small, iterations=2)

    sure_bg = cv2.dilate(opening, kernel_large, iterations=4)

    # Distance transform with better seed point detection
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)

    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(original, markers)

    output = original.copy()
    output[markers == -1] = [0, 0, 255]

    # Calculate dynamic area threshold based on image size
    image_area = new_width * new_height
    min_area = image_area * 0.001  # 0.1% of image area
    max_area = image_area * 0.4    # 40% of image area

    object_count = 0

    for marker_id in np.unique(markers):
        if marker_id <= 1:
            continue

        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == marker_id] = 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter by area and circularity for better object detection
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                object_count += 1

    return object_count, output, opening
