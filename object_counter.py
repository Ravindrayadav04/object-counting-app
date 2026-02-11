import cv2
import numpy as np


def get_main_stack_roi(img):
    """Find the main cloth stack region (largest contour)"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)

    # Threshold saturation
    _, mask = cv2.threshold(s, 50, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None, None

    # largest contour = main stack
    largest = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest)

    return (x, y, w, h), mask


def count_objects_watershed(roi_img):
    """Watershed segmentation inside ROI"""
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

    # Heavy blur removes embroidery
    blur = cv2.GaussianBlur(gray, (11, 11), 0)

    # Otsu threshold
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)

    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    _, sure_fg = cv2.threshold(dist_transform, 0.35 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    num_labels, markers = cv2.connectedComponents(sure_fg)

    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(roi_img, markers)

    unique_markers = np.unique(markers)

    detected = 0
    boxes = []

    for marker_id in unique_markers:
        if marker_id <= 1:
            continue

        mask = np.uint8(markers == marker_id) * 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # filter noise
            if area < 1500:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # avoid tiny cuts
            if w < 40 or h < 40:
                continue

            detected += 1
            boxes.append((x, y, w, h))

    return detected, thresh, boxes


def count_cloth_stacks(image):
    img = image.copy()

    # resize for stable detection
    h, w = img.shape[:2]
    new_w = 1000
    ratio = new_w / w
    img = cv2.resize(img, (new_w, int(h * ratio)))

    output_img = img.copy()

    roi_rect, roi_mask = get_main_stack_roi(img)

    if roi_rect is None:
        return 0, roi_mask, output_img

    x, y, rw, rh = roi_rect

    roi_img = img[y:y+rh, x:x+rw]

    count, processed_mask, boxes = count_objects_watershed(roi_img)

    # Draw ROI box
    cv2.rectangle(output_img, (x, y), (x+rw, y+rh), (255, 0, 0), 3)

    # Draw detected cloth boxes
    for (bx, by, bw, bh) in boxes:
        cv2.rectangle(output_img, (x+bx, y+by), (x+bx+bw, y+by+bh), (0, 255, 0), 3)

    return count, processed_mask, output_img
