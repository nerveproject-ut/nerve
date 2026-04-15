#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Distance-aware Mosaic Detection for YOLOX.

This version preserves the distance column (6th column) in labels
when training with distance estimation.
"""

import random
import cv2
import numpy as np

from yolox.utils import adjust_box_anns, get_local_rank
from ..data_augment import random_affine
from .datasets_wrapper import Dataset
from .mosaicdetection import get_mosaic_coordinate


class MosaicDetectionWithDistance(Dataset):
    """
    Detection dataset wrapper that performs mosaic/mixup augmentation
    while preserving distance labels.
    
    Handles labels with 6 columns: [x1, y1, x2, y2, class_id, distance]
    
    IMPORTANT: Always outputs labels with consistent number of columns
    (6 for distance mode, 5 for standard mode) to ensure proper batching.
    """

    def __init__(
        self, dataset, img_size, mosaic=True, preproc=None,
        degrees=10.0, translate=0.1, mosaic_scale=(0.5, 1.5),
        mixup_scale=(0.5, 1.5), shear=2.0, enable_mixup=True,
        mosaic_prob=1.0, mixup_prob=1.0, include_distance=True, *args
    ):
        """
        Args:
            dataset(Dataset): Pytorch dataset object.
            img_size (tuple): Target image size.
            mosaic (bool): Enable mosaic augmentation.
            preproc (func): Preprocessing function.
            degrees (float): Rotation degrees for affine transform.
            translate (float): Translation for affine transform.
            mosaic_scale (tuple): Scale range for mosaic.
            mixup_scale (tuple): Scale range for mixup.
            shear (float): Shear for affine transform.
            enable_mixup (bool): Enable mixup augmentation.
            mosaic_prob (float): Probability of mosaic augmentation.
            mixup_prob (float): Probability of mixup augmentation.
            include_distance (bool): Whether labels include distance column.
        """
        super().__init__(img_size, mosaic=mosaic)
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = mosaic_scale
        self.shear = shear
        self.mixup_scale = mixup_scale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.local_rank = get_local_rank()
        self.include_distance = include_distance
        # Fixed number of columns for consistent output
        self.num_label_cols = 6 if include_distance else 5

    def __len__(self):
        return len(self._dataset)

    def _ensure_label_cols(self, labels):
        """
        Ensure labels have the correct number of columns.
        Pads with zeros if needed (e.g., for distance column).
        """
        if len(labels) == 0:
            return np.zeros((0, self.num_label_cols), dtype=np.float32)
        
        if labels.shape[1] < self.num_label_cols:
            # Pad with zeros for missing columns
            padding = np.zeros((labels.shape[0], self.num_label_cols - labels.shape[1]))
            labels = np.hstack((labels, padding))
        elif labels.shape[1] > self.num_label_cols:
            # Truncate extra columns
            labels = labels[:, :self.num_label_cols]
        
        return labels.astype(np.float32)

    @Dataset.mosaic_getitem
    def __getitem__(self, idx):
        if self.enable_mosaic and random.random() < self.mosaic_prob:
            mosaic_labels = []
            input_dim = self._dataset.input_dim
            input_h, input_w = input_dim[0], input_dim[1]

            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            # 3 additional image indices
            indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]

            for i_mosaic, index in enumerate(indices):
                img, _labels, _, img_id = self._dataset.pull_item(index)
                
                # Ensure consistent column count
                _labels = self._ensure_label_cols(_labels)
                
                h0, w0 = img.shape[:2]
                scale = min(1. * input_h / h0, 1. * input_w / w0)
                img = cv2.resize(
                    img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
                )
                (h, w, c) = img.shape[:3]
                if i_mosaic == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)

                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                    mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
                )

                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                labels = _labels.copy()
                if len(_labels) > 0:
                    # Transform bbox coordinates (columns 0-3)
                    labels[:, 0] = scale * _labels[:, 0] + padw
                    labels[:, 1] = scale * _labels[:, 1] + padh
                    labels[:, 2] = scale * _labels[:, 2] + padw
                    labels[:, 3] = scale * _labels[:, 3] + padh
                    # Note: columns 4 (class) and 5 (distance) remain unchanged
                mosaic_labels.append(labels)

            # Concatenate all labels
            if len(mosaic_labels) > 0:
                # Filter out empty arrays before concatenation
                non_empty = [lb for lb in mosaic_labels if len(lb) > 0]
                if len(non_empty) > 0:
                    mosaic_labels = np.concatenate(non_empty, 0)
                    np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
                    np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
                    np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
                    np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])
                else:
                    # All images had no labels - create empty array with correct shape
                    mosaic_labels = np.zeros((0, self.num_label_cols), dtype=np.float32)
            else:
                mosaic_labels = np.zeros((0, self.num_label_cols), dtype=np.float32)

            mosaic_img, mosaic_labels = random_affine(
                mosaic_img,
                mosaic_labels,
                target_size=(input_w, input_h),
                degrees=self.degrees,
                translate=self.translate,
                scales=self.scale,
                shear=self.shear,
            )

            # Mixup augmentation (preserves distance)
            if (
                self.enable_mixup
                and len(mosaic_labels) > 0
                and random.random() < self.mixup_prob
            ):
                mosaic_img, mosaic_labels = self.mixup(mosaic_img, mosaic_labels, self.input_dim)
            
            mix_img, padded_labels = self.preproc(mosaic_img, mosaic_labels, self.input_dim)
            img_info = (mix_img.shape[1], mix_img.shape[0])

            return mix_img, padded_labels, img_info, img_id

        else:
            self._dataset._input_dim = self.input_dim
            img, label, img_info, img_id = self._dataset.pull_item(idx)
            # Ensure consistent column count before preprocessing
            label = self._ensure_label_cols(label)
            img, label = self.preproc(img, label, self.input_dim)
            return img, label, img_info, img_id

    def mixup(self, origin_img, origin_labels, input_dim):
        """
        Mixup augmentation that preserves distance labels.
        
        Handles labels with consistent column count.
        """
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        cp_labels = []
        while len(cp_labels) == 0:
            cp_index = random.randint(0, self.__len__() - 1)
            cp_labels = self._dataset.load_anno(cp_index)
        img, cp_labels, _, _ = self._dataset.pull_item(cp_index)
        
        # Ensure consistent column count
        cp_labels = self._ensure_label_cols(cp_labels)

        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114
        else:
            cp_img = np.ones(input_dim, dtype=np.uint8) * 114

        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )

        cp_img[
            : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_img

        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor

        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
        )
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
            y_offset: y_offset + target_h, x_offset: x_offset + target_w
        ]

        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
            )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )

        # Preserve all columns beyond bbox (class and distance)
        extra_labels = cp_labels[:, 4:].copy()
        box_labels = cp_bboxes_transformed_np
        labels = np.hstack((box_labels, extra_labels))
        
        # Ensure both have same columns (should already be guaranteed)
        labels = self._ensure_label_cols(labels)
        origin_labels = self._ensure_label_cols(origin_labels)
        
        origin_labels = np.vstack((origin_labels, labels))
        origin_img = origin_img.astype(np.float32)
        origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

        return origin_img.astype(np.uint8), origin_labels
