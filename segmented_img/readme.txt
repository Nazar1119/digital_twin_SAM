folder name - x_y_z_r_e
x - pred_iou_thresh(sets the minimum predicted IoU (Intersection over Union) that a mask must have to be considered valid)
y - stability_score_thresh(this threshold filters masks based on their stability score, which measures consistency across different IoU thresholds)
z - min_aspect_ratio(parameter filters masks based on their aspect ratio (minimum of width/height or height/width))
r - min_iou_with_largest(parameter checks the IoU between a mask and all larger masks (sorted by area))
e - iou_threshold(parameter determines when to merge overlapping masks. If the IoU between two masks is greater than 'e', they are combined into a single mask)