import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys

sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def filter_sinter_stones(masks, area_range, min_aspect_ratio, min_iou_with_largest):
    filtered_masks = []
    if not masks:
        return filtered_masks
    
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    for i, mask in enumerate(sorted_masks):
        area = mask['area']
        bbox = mask['bbox']
        width = bbox[2]
        height = bbox[3]
        aspect_ratio = min(width / height, height / width)
        
        is_redundant = False
        for larger_mask in sorted_masks[:i]:
            larger_seg = larger_mask['segmentation']
            current_seg = mask['segmentation']
            intersection = np.logical_and(larger_seg, current_seg).sum()
            union = np.logical_or(larger_seg, current_seg).sum()
            iou = intersection / union if union > 0 else 0
            if iou > min_iou_with_largest:
                is_redundant = True
                break
        
        if (area_range[0] <= area <= area_range[1] and 
            aspect_ratio >= min_aspect_ratio and 
            not is_redundant):
            filtered_masks.append(mask)
    
    return filtered_masks

def merge_overlapping_masks(masks, iou_threshold):
    if not masks:
        return masks
    
    merged_masks = []
    remaining_masks = masks.copy()
    
    while remaining_masks:
        current_mask = remaining_masks.pop(0)
        current_seg = current_mask['segmentation']
        current_bbox = current_mask['bbox']
        
        overlapping = []
        for i, other_mask in enumerate(remaining_masks):
            other_seg = other_mask['segmentation']
            intersection = np.logical_and(current_seg, other_seg).sum()
            union = np.logical_or(current_seg, other_seg).sum()
            iou = intersection / union if union > 0 else 0
            
            if iou > iou_threshold:
                overlapping.append(i)
        
        for idx in sorted(overlapping, reverse=True):
            other_mask = remaining_masks.pop(idx)
            current_seg = np.logical_or(current_seg, other_mask['segmentation'])
            other_bbox = other_mask['bbox']
            current_bbox = [
                min(current_bbox[0], other_bbox[0]),
                min(current_bbox[1], other_bbox[1]),
                max(current_bbox[0] + current_bbox[2], other_bbox[0] + other_bbox[2]) - min(current_bbox[0], other_bbox[0]),
                max(current_bbox[1] + current_bbox[3], other_bbox[1] + other_bbox[3]) - min(current_bbox[1], other_bbox[1])
            ]
        
        current_mask['segmentation'] = current_seg
        current_mask['bbox'] = current_bbox
        merged_masks.append(current_mask)
    
    return merged_masks

def exclude_text_regions(masks, image_shape, text_regions):
    height, width = image_shape[:2]
    filtered_masks = []
    
    for mask in masks:
        m = mask['segmentation']
        mask_height, mask_width = m.shape
        overlaps_text = False
        
        for x, y, w, h in text_regions:
            mask_x, mask_y = int(x * mask_width / width), int(y * mask_height / height)
            mask_w, mask_h = int(w * mask_width / width), int(h * mask_height / height)
            
            if np.any(m[max(0, mask_y):min(mask_height, mask_y + mask_h),
                        max(0, mask_x):min(mask_width, mask_x + mask_w)]):
                overlaps_text = True
                break
        
        if not overlaps_text:
            filtered_masks.append(mask)
    
    return filtered_masks

def color_segmentation(masks, base_image):
    segmented_image = base_image.copy()
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]]
    
    for i, mask in enumerate(masks):
        m = mask['segmentation']
        m_resized = cv2.resize(m.astype(np.uint8), (segmented_image.shape[1], segmented_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        color = colors[i % len(colors)]
        color_mask = np.zeros_like(segmented_image, dtype=np.float32)
        
        for c in range(3):
            color_mask[:, :, c] = m_resized * color[c]
        color_mask = np.uint8(color_mask * 0.3 * 255)
        segmented_image = cv2.addWeighted(segmented_image, 1, color_mask, 0.7, 0)
    
    return segmented_image

def draw_bounding_boxes(image, masks, color=(0, 255, 0), thickness=1):
    result_image = image.copy()
    processed_bboxes = set()
    
    for mask in masks:
        bbox = mask['bbox']
        bbox_tuple = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        if bbox_tuple not in processed_bboxes:
            x_min, y_min, w, h = bbox_tuple
            cv2.rectangle(result_image, (x_min, y_min), (x_min + w, y_min + h), color, thickness)
            processed_bboxes.add(bbox_tuple)
    
    return result_image

def measure_box_dimensions(masks):
    print("\nBounding Box Dimensions (width, height) in pixels:")
    for i, mask in enumerate(masks, 1):
        bbox = mask['bbox']
        width = bbox[2]  # Width in pixels
        height = bbox[3]  # Height in pixels
        print(f"Box {i}: Width = {width} pixels, Height = {height} pixels")

image = cv2.imread('/home/nt646jh/directory/folder/bc_nazarii_tymochko/img1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# SAM setup
sam_checkpoint = '/home/nt646jh/directory/folder/bc_nazarii_tymochko/SegmentAnything/sam_vit_h_4b8939.pth'
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# Mask generation with adjusted parameters (keeping your specified values but increasing min_mask_region_area)
mask_generator1_ = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=64,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.7,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=50,  # Increased from 1 to reduce small masks
)

image_original = cv2.imread('/home/nt646jh/directory/folder/bc_nazarii_tymochko/img1.jpg')
image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)

#Default filters:

image = cv2.GaussianBlur(image_original, (5, 5), 0)
clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
image = np.stack([clahe.apply(image[:, :, i]) for i in range(3)], axis=2)
image_resized = cv2.resize(image, (1024, 768))


# Filter: Zhang et al. used image enhancement techniques to improve the quality of captured images, 
# making edges and particle boundaries more distinguishable for subsequent edge detection. 
# This likely involved adjusting contrast, brightness, or applying filters to reduce noise while preserving details.

# image = cv2.cvtColor(image_original, cv2.COLOR_RGB2GRAY)  
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  
# image = clahe.apply(image)
# image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) 
# image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)  
# image_resized = cv2.resize(image, (1024, 768))

masks_original = mask_generator1_.generate(image_resized)
print(f"Number of masks generated: {len(masks_original)}")

text_regions = [
    (0, 0, 310, 50),  # Top-left text
    (image_resized.shape[1] - 63 - 208, image_resized.shape[0] - 72 - (27 // 2), 208, 25)  # Bottom-right text (exact size and position)
]

filtered_masks = exclude_text_regions(masks_original, image_resized.shape, text_regions)
# sinter_stone_masks = filter_sinter_stones(filtered_masks, area_range=(10, 3500), min_aspect_ratio=0.7, min_iou_with_largest=0.5)
# sinter_stone_masks = merge_overlapping_masks(sinter_stone_masks, iou_threshold=0.5)

sinter_stone_masks = filter_sinter_stones(filtered_masks, area_range=(10, 5000), min_aspect_ratio=0.8, min_iou_with_largest=0.7)
sinter_stone_masks = merge_overlapping_masks(sinter_stone_masks, iou_threshold=0.6)

measure_box_dimensions(sinter_stone_masks)

# Generate segmented image and add bounding boxes

# # Display and save results
# plt.figure(figsize=(50, 25))
# plt.imshow(segmented_image_with_boxes)
# plt.axis('off')
# plt.savefig('/home/nt646jh/directory/folder/bc_nazarii_tymochko/64_filter1.png')
# plt.close()


# Extract bounding box dimensions and combine them into labels
# Extract bounding box dimensions
widths = [mask['bbox'][2] for mask in sinter_stone_masks]  # Widths in pixels
heights = [mask['bbox'][3] for mask in sinter_stone_masks]  # Heights in pixels

# Define bins for widths (e.g., 0-10, 10-20, ..., 100-110, etc.)
bin_size = 10  # Adjust this based on your data
max_width = max(widths)
bins = np.arange(0, max_width + bin_size, bin_size)
width_labels = [f"{int(b)}-{int(b + bin_size)}" for b in bins[:-1]]

# Bin the widths
binned_widths = np.digitize(widths, bins, right=True)
binned_counts = np.bincount(binned_widths)[1:]  # Ignore the 0th bin (below the first bin)

# Create histogram
plt.figure(figsize=(12, 6))
plt.bar(width_labels, binned_counts, color='purple', alpha=0.7, edgecolor='black')
plt.title('Histogram of Bounding Box Widths (Binned)')
plt.xlabel('Width Range (pixels)')
plt.ylabel('Frequency')

# Rotate x-axis labels for readability
plt.xticks(rotation=45, ha='right')

# Adjust layout and save histogram
plt.tight_layout()
plt.savefig('/home/nt646jh/directory/folder/bc_nazarii_tymochko/64_filter1_histogram_binned_width.png')
plt.close()