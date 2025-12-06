"""
Automatic Hangul Character Image Segmentation.

This script implements automatic segmentation of complete Hangul character images
into cho/jung/jong regions for Jamo-level classification.
"""

import os
import glob
import cv2
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt


def detect_components(img: np.ndarray, debug=False) -> List[Tuple[int, int, int, int]]:
    """
    Detect individual components (strokes) in a Hangul character image.
    
    Args:
        img: Grayscale image (64x64)
        debug: Whether to show debug visualization
    
    Returns:
        List of bounding boxes [(x, y, w, h), ...]
    """
    # Threshold to binary
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get bounding boxes
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 3 and h > 3:  # Filter noise
            boxes.append((x, y, w, h))
    
    # Sort by position (left to right, top to bottom)
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    
    if debug:
        debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for i, (x, y, w, h) in enumerate(boxes):
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.putText(debug_img, str(i), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        plt.imshow(debug_img)
        plt.title(f"Components: {len(boxes)}")
        plt.axis('off')
        plt.show()
    
    return boxes


def segment_hangul_fixed_regions(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Segment Hangul character using fixed region approach.
    
    Simple heuristic:
    - Cho (Initial): Left 50% or Top 50% (depending on layout)
    - Jung (Medial): Right 50% or Middle
    - Jong (Final): Bottom portion
    
    Args:
        img: Grayscale image (64x64)
    
    Returns:
        (cho_region, jung_region, jong_region) as 64x64 images
    """
    h, w = img.shape
    
    # Approach 1: Simple left-right split (works for many characters)
    mid_w = w // 2
    
    # Cho: Left half
    cho_region = img[:, :mid_w]
    cho_region = cv2.resize(cho_region, (64, 64))
    
    # Jung: Right upper half
    jung_region = img[:h*2//3, mid_w:]
    jung_region = cv2.resize(jung_region, (64, 64))
    
    # Jong: Bottom portion
    jong_region = img[h*2//3:, :]
    jong_region = cv2.resize(jong_region, (64, 64))
    
    return cho_region, jung_region, jong_region


def segment_hangul_adaptive(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Adaptive segmentation based on detected components.
    
    Strategy:
    1. Detect all stroke components
    2. Group into 2-3 regions based on position
    3. Assign to cho/jung/jong based on Hangul structure rules
    
    Args:
        img: Grayscale image (64x64)
    
    Returns:
        (cho_region, jung_region, jong_region) as 64x64 images
    """
    boxes = detect_components(img, debug=False)
    h, w = img.shape
    
    if len(boxes) == 0:
        # No components detected, return full image for all
        return img.copy(), img.copy(), np.zeros((64, 64), dtype=np.uint8)
    
    # Analyze layout
    avg_x = np.mean([b[0] + b[2]/2 for b in boxes])
    avg_y = np.mean([b[1] + b[3]/2 for b in boxes])
    
    # Determine if horizontal or vertical layout
    x_spread = max([b[0] + b[2] for b in boxes]) - min([b[0] for b in boxes])
    y_spread = max([b[1] + b[3] for b in boxes]) - min([b[1] for b in boxes])
    
    is_horizontal_layout = x_spread > y_spread
    
    if is_horizontal_layout:
        # Left-right layout (ㄱ+ㅏ style)
        mid_x = w // 2
        cho_region = img[:, :mid_x]
        jung_region = img[:h*2//3, mid_x:]
        jong_region = img[h*2//3:, :]
    else:
        # Top-bottom layout (ㅎ+ㅗ style)
        mid_y = h // 2
        cho_region = img[:mid_y, :]
        jung_region = img[mid_y:h*2//3, :]
        jong_region = img[h*2//3:, :]
    
    # Resize to standard size
    cho_region = cv2.resize(cho_region, (64, 64))
    jung_region = cv2.resize(jung_region, (64, 64))
    jong_region = cv2.resize(jong_region, (64, 64))
    
    return cho_region, jung_region, jong_region


if __name__ == "__main__":
    # Test with sample images
    import glob
    
    print("Testing segmentation on sample images...")
    
    # Load a few sample images
    sample_dir = "archive/Hangul Database/Hangul Database"
    classes = ["jeong", "ga", "gong", "i"]
    
    fig, axes = plt.subplots(len(classes), 4, figsize=(16, 4*len(classes)))
    
    for row_idx, class_name in enumerate(classes):
        class_dir = os.path.join(sample_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        
        files = glob.glob(os.path.join(class_dir, "*.jpg"))
        if len(files) == 0:
            continue
        
        # Load first image
        img = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        
        # Segment
        cho, jung, jong = segment_hangul_adaptive(img)
        
        # Display
        axes[row_idx, 0].imshow(img, cmap='gray')
        axes[row_idx, 0].set_title(f"Original: {class_name}")
        axes[row_idx, 0].axis('off')
        
        axes[row_idx, 1].imshow(cho, cmap='gray')
        axes[row_idx, 1].set_title("Cho (초성)")
        axes[row_idx, 1].axis('off')
        
        axes[row_idx, 2].imshow(jung, cmap='gray')
        axes[row_idx, 2].set_title("Jung (중성)")
        axes[row_idx, 2].axis('off')
        
        axes[row_idx, 3].imshow(jong, cmap='gray')
        axes[row_idx, 3].set_title("Jong (종성)")
        axes[row_idx, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig("results/jamo_segmentation_test.png", dpi=150)
    print("Saved: results/jamo_segmentation_test.png")
    plt.close()

