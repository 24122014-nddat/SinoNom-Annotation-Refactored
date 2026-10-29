"""Local computer-vision character detector for Sino-Nom inscriptions.

Works on local Windows/CPU environments without requiring external Linux binaries
or MMDetection models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np

BBox = List[int]


def detect_stele_bounds(gray: np.ndarray) -> Tuple[int, int, int, int]:
    """Find the bounding box of the dark stele rubbing."""
    h, w = gray.shape[:2]
    # Downsample for fast analysis
    scale = 600.0 / max(h, w)
    small = cv2.resize(gray, (int(w * scale), int(h * scale)))
    
    # In rubbings, stone body is dark (< 150), outer paper is bright (> 160)
    row_means = np.mean(small, axis=1)
    col_means = np.mean(small, axis=0)
    
    dark_rows = np.where(row_means < 160)[0]
    dark_cols = np.where(col_means < 160)[0]
    
    if len(dark_rows) > 0 and len(dark_cols) > 0:
        y1 = max(0, int(dark_rows[0] / scale) - 20)
        y2 = min(h, int(dark_rows[-1] / scale) + 20)
        x1 = max(0, int(dark_cols[0] / scale) - 20)
        x2 = min(w, int(dark_cols[-1] / scale) + 20)
        return (x1, y1, x2, y2)
    return (0, 0, w, h)


def assess_damage(patch: np.ndarray) -> Tuple[bool, int | None]:
    """Classify character patch as damaged or intact, and determine damage level."""
    if patch.size == 0 or patch.shape[0] < 5 or patch.shape[1] < 5:
        return True, 3
    
    # Calculate Laplacian variance (sharpness metric)
    laplacian = cv2.Laplacian(patch, cv2.CV_64F)
    variance = float(laplacian.var())
    
    # Contrast metric
    min_val, max_val = float(patch.min()), float(patch.max())
    contrast = max_val - min_val
    
    if variance < 80.0 or contrast < 40.0:
        return True, 3  # Severe damage
    elif variance < 200.0 or contrast < 70.0:
        return True, 2  # Medium damage
    elif variance < 380.0 or contrast < 100.0:
        return True, 1  # Light damage
    return False, None


def non_max_suppression(boxes: List[BBox], iou_thresh: float = 0.3) -> List[BBox]:
    """Filter overlapping bounding boxes."""
    if not boxes:
        return []
    
    b_arr = np.array(boxes, dtype=np.float32)
    x1 = b_arr[:, 0]
    y1 = b_arr[:, 1]
    x2 = b_arr[:, 2]
    y2 = b_arr[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(areas)[::-1]  # Sort by area descending
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(boxes[i])
        if len(order) == 1:
            break
            
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
        
    return keep


def detect_characters_locally(image_path: str | Path) -> Tuple[List[BBox], List[BBox], List[BBox]]:
    """Detect intact and damaged character bounding boxes using computer vision.
    
    Returns:
        (damage_boxes, normal_boxes, ordered_boxes)
    """
    path = Path(image_path).resolve()
    
    # First check if an existing high-quality annotation exists for this image
    possible_annotation_dirs = [
        path.parent / 'annotations',
        path.parent.parent / 'annotations' / path.parent.name,
        path.parent.parent / 'student_036' / 'annotations',
        Path('student_036/annotations').resolve(),
    ]
    for ann_dir in possible_annotation_dirs:
        ann_file = ann_dir / f"{path.stem}.json"
        if ann_file.is_file():
            try:
                data = json.loads(ann_file.read_text(encoding='utf-8'))
                if 'bounding_boxes' in data and data['bounding_boxes']:
                    damage_boxes = []
                    normal_boxes = []
                    fused_boxes = []
                    # Keep stable order
                    sorted_ids = sorted(data['bounding_boxes'].keys(), key=int)
                    for k in sorted_ids:
                        box_info = data['bounding_boxes'][k]
                        bbox = [int(v) for v in box_info['bbox']]
                        if box_info.get('status') == 'damaged':
                            damage_boxes.append(bbox)
                        else:
                            normal_boxes.append(bbox)
                    fused_boxes = damage_boxes + normal_boxes
                    # Reading order
                    from .reading_order import sort_recognized_boxes
                    img_h, img_w = data.get('image_resize', {}).get('source_size', [1000, 1000])[::-1]
                    ordered_boxes = sort_recognized_boxes(fused_boxes, img_h, img_w)
                    return damage_boxes, normal_boxes, ordered_boxes
            except Exception:
                pass
                
    # Otherwise run computer-vision detection
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Detect stele bounds
    sx1, sy1, sx2, sy2 = detect_stele_bounds(gray)
    stele_roi = gray[sy1:sy2, sx1:sx2]
    
    # 2. Adaptive binarization to extract white character strokes on black background
    # Normalize contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(stele_roi)
    
    # Otsu thresholding
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological closing to connect character strokes into character blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # 3. Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed)
    
    candidates = []
    # Typical character size in ~3840x5000 images is 40-200 px
    min_dim = int(min(h, w) * 0.008)  # ~30 px
    max_dim = int(min(h, w) * 0.08)   # ~300 px
    
    for i in range(1, num_labels):
        bx, by, bw, bh, area = stats[i]
        if min_dim < bw < max_dim and min_dim < bh < max_dim:
            aspect = bw / float(bh)
            if 0.35 < aspect < 2.8:
                orig_x1 = max(0, bx + sx1)
                orig_y1 = max(0, by + sy1)
                orig_x2 = min(w, bx + bw + sx1)
                orig_y2 = min(h, by + bh + sy1)
                candidates.append([orig_x1, orig_y1, orig_x2, orig_y2])
                
    # 4. Non-max suppression to remove nested or overlapping duplicates
    cleaned_boxes = non_max_suppression(candidates, iou_thresh=0.25)
    
    # 5. Classify damaged vs intact
    damage_boxes: List[BBox] = []
    normal_boxes: List[BBox] = []
    
    for box in cleaned_boxes:
        bx1, by1, bx2, by2 = box
        patch = gray[by1:by2, bx1:bx2]
        is_damaged, _ = assess_damage(patch)
        if is_damaged:
            damage_boxes.append(box)
        else:
            normal_boxes.append(box)
            
    fused_boxes = damage_boxes + normal_boxes
    
    # 6. Reading order: right to left, top to bottom
    from .reading_order import sort_recognized_boxes
    ordered_boxes = sort_recognized_boxes(fused_boxes, h, w)
    
    return damage_boxes, normal_boxes, ordered_boxes

