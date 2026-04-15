#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Distance-aware data augmentation for YOLOX.

These transforms preserve the distance column (6th column) in labels
when training with distance estimation.
"""

import random
import cv2
import numpy as np

from yolox.utils import xyxy2cxcywh
from .data_augment import augment_hsv, preproc, _mirror


class TrainTransformWithDistance:
    """
    Training transform that preserves the distance column.
    
    Handles labels with 6 columns: [x1, y1, x2, y2, class_id, distance]
    Output format: [class_id, cx, cy, w, h, distance] (6 columns)
    
    IMPORTANT: Always outputs 6 columns to ensure consistent batch shapes.
    """
    
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0, include_distance=True):
        """
        Args:
            max_labels: Maximum number of labels per image
            flip_prob: Probability of horizontal flip
            hsv_prob: Probability of HSV augmentation
            include_distance: If True, always output 6 columns. 
                              Must be True for distance estimation training.
        """
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.include_distance = include_distance
        # Fixed output columns for consistent batch shapes
        self.num_cols = 6 if include_distance else 5

    def __call__(self, image, targets, input_dim):
        """
        Apply transforms to image and targets.
        
        Args:
            image: Input image
            targets: Labels with shape (N, 5) or (N, 6)
                     Format: [x1, y1, x2, y2, class_id] or [x1, y1, x2, y2, class_id, distance]
            input_dim: Target input dimensions
            
        Returns:
            Transformed image and labels with shape (max_labels, num_cols)
            Output labels format: [class_id, cx, cy, w, h] or [class_id, cx, cy, w, h, distance]
        """
        # Check if input has distance (may vary per sample, but output is fixed)
        input_has_distance = targets.shape[1] >= 6 if len(targets) > 0 else False
        
        boxes = targets[:, :4].copy() if len(targets) > 0 else np.zeros((0, 4))
        labels = targets[:, 4].copy() if len(targets) > 0 else np.zeros((0,))
        
        # Get distance if available in input
        if input_has_distance:
            distances = targets[:, 5].copy()
        else:
            # No distance in input - use zeros
            distances = np.zeros(len(targets))
        
        if len(boxes) == 0:
            # Return empty labels with CONSISTENT shape
            targets = np.zeros((self.max_labels, self.num_cols), dtype=np.float32)
            image, r_o = preproc(image, input_dim)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        if input_has_distance:
            distances_o = targets_o[:, 5]
        else:
            distances_o = np.zeros(len(targets_o))
        
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)

        if random.random() < self.hsv_prob:
            augment_hsv(image)
        
        image_t, boxes = _mirror(image, boxes, self.flip_prob)
        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim)
        
        # boxes [xyxy] to [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]
        distances_t = distances[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o
            distances_t = distances_o

        labels_t = np.expand_dims(labels_t, 1)
        
        # Create output: [class, cx, cy, w, h, (distance)]
        if self.include_distance:
            distances_t = np.expand_dims(distances_t, 1)
            targets_t = np.hstack((labels_t, boxes_t, distances_t))
        else:
            targets_t = np.hstack((labels_t, boxes_t))
        
        # ALWAYS use fixed num_cols for consistent batch shapes
        padded_labels = np.zeros((self.max_labels, self.num_cols))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image_t, padded_labels


class ValTransformWithDistance:
    """
    Validation transform that handles distance labels.
    """

    def __init__(self, swap=(2, 0, 1), legacy=False, include_distance=True):
        self.swap = swap
        self.legacy = legacy
        self.include_distance = include_distance
        self.num_cols = 6 if include_distance else 5

    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        
        return img, np.zeros((1, self.num_cols))
