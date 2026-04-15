#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import io
import itertools
import json
import tempfile
import time
from collections import ChainMap, defaultdict
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm

import numpy as np

import torch

from yolox.data.datasets import COCO_CLASSES
from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)


def per_class_AR_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AR"], colums=6):
    per_class_AR = {}
    recalls = coco_eval.eval["recall"]
    # dimension of recalls: [TxKxAxM]
    # recall has dims (iou, cls, area range, max dets)
    assert len(class_names) == recalls.shape[1]

    for idx, name in enumerate(class_names):
        recall = recalls[:, idx, 0, -1]
        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")
        per_class_AR[name] = float(ar * 100)

    num_cols = min(colums, len(per_class_AR) * len(headers))
    result_pair = [x for pair in per_class_AR.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


def per_class_AP_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AP"], colums=6):
    per_class_AP = {}
    precisions = coco_eval.eval["precision"]
    # dimension of precisions: [TxRxKxAxM]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        per_class_AP[name] = float(ap * 100)

    num_cols = min(colums, len(per_class_AP) * len(headers))
    result_pair = [x for pair in per_class_AP.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


class COCOEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self,
        dataloader,
        img_size: int,
        confthre: float,
        nmsthre: float,
        num_classes: int,
        testdev: bool = False,
        per_class_AP: bool = True,
        per_class_AR: bool = True,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size: image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre: confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre: IoU threshold of non-max supression ranging from 0 to 1.
            per_class_AP: Show per class AP during evalution or not. Default to True.
            per_class_AR: Show per class AR during evalution or not. Default to True.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev
        self.per_class_AP = per_class_AP
        self.per_class_AR = per_class_AR
        
        # Visualization settings
        self.save_dir = None
        self.class_names = None
        self._val_batches_saved = 0
        self._max_val_batches = 3
        
        # Confusion matrix data collection
        self._confusion_matrix_data = {
            'detections': [],  # List of detection arrays per image
            'labels': [],      # List of ground truth arrays per image
        }

    def enable_visualization(self, save_dir, class_names=None):
        """
        Enable validation batch visualization during evaluation.
        
        Args:
            save_dir: Directory to save validation batch visualizations
            class_names: List of class names for labeling
        """
        self.save_dir = save_dir
        self.class_names = class_names
        self._val_batches_saved = 0
        # Reset confusion matrix data
        self._confusion_matrix_data = {
            'detections': [],
            'labels': [],
        }
    
    def _collect_confusion_matrix_data(self, outputs, ids, info_imgs, img_h, img_w):
        """
        Collect detection and ground truth data for confusion matrix computation.
        
        Args:
            outputs: List of prediction outputs per image
            ids: Image IDs for fetching ground truth from COCO API
            info_imgs: Image info (heights, widths)
            img_h, img_w: Model input image dimensions
        """
        try:
            coco = self.dataloader.dataset.coco
            
            # Get ground truth for each image
            ids_np = ids.cpu().numpy().flatten() if hasattr(ids, 'cpu') else np.array(ids).flatten()
            orig_heights = info_imgs[0].cpu().numpy() if hasattr(info_imgs[0], 'cpu') else np.array(info_imgs[0])
            orig_widths = info_imgs[1].cpu().numpy() if hasattr(info_imgs[1], 'cpu') else np.array(info_imgs[1])
            
            for idx, img_id in enumerate(ids_np):
                img_id = int(img_id)
                
                # Get ground truth
                ann_ids = coco.getAnnIds(imgIds=img_id)
                annotations = coco.loadAnns(ann_ids)
                
                orig_h = float(orig_heights[idx]) if idx < len(orig_heights) else float(img_h)
                orig_w = float(orig_widths[idx]) if idx < len(orig_widths) else float(img_w)
                scale = min(img_h / orig_h, img_w / orig_w)
                
                # Ground truth labels: [class, x1, y1, x2, y2] in scaled coords
                gt_labels = []
                for ann in annotations:
                    x, y, w, h = ann['bbox']
                    cat_id = ann['category_id']
                    
                    # Map category ID to class index
                    cls = cat_id
                    if hasattr(self.dataloader.dataset, 'class_ids'):
                        class_ids = self.dataloader.dataset.class_ids
                        if cat_id in class_ids:
                            cls = class_ids.index(cat_id)
                        else:
                            continue
                    
                    # Scale to model input size
                    x1 = x * scale
                    y1 = y * scale
                    x2 = (x + w) * scale
                    y2 = (y + h) * scale
                    
                    gt_labels.append([cls, x1, y1, x2, y2])
                
                # Get detections: [x1, y1, x2, y2, conf, class]
                det_labels = []
                if outputs is not None and idx < len(outputs) and outputs[idx] is not None:
                    output = outputs[idx]
                    if hasattr(output, 'cpu'):
                        output = output.cpu().numpy()
                    
                    for det in output:
                        x1, y1, x2, y2 = det[:4]
                        conf = float(det[4]) * float(det[5])  # obj_conf * cls_conf
                        cls = int(det[6])
                        det_labels.append([x1, y1, x2, y2, conf, cls])
                
                self._confusion_matrix_data['labels'].append(
                    np.array(gt_labels) if gt_labels else np.zeros((0, 5))
                )
                self._confusion_matrix_data['detections'].append(
                    np.array(det_labels) if det_labels else np.zeros((0, 6))
                )
                
        except Exception as e:
            # Silently ignore errors in confusion matrix collection
            pass
    
    def get_confusion_matrix_data(self):
        """
        Get collected confusion matrix data.
        
        Returns:
            Dict with 'detections' and 'labels' lists
        """
        return self._confusion_matrix_data
    
    def _save_val_batch_visualization(self, imgs, outputs, ids, batch_idx, info_imgs=None):
        """
        Save validation batch visualization (labels and predictions).
        
        Args:
            imgs: Batch of images tensor (N, C, H, W)
            outputs: List of prediction outputs per image (each: [x1, y1, x2, y2, obj_conf, cls_conf, cls])
            ids: Image IDs for fetching ground truth from COCO API
            batch_idx: Current batch index
            info_imgs: Image info (heights, widths) for coordinate scaling
        """
        if self.save_dir is None or batch_idx >= self._max_val_batches:
            return
        
        if self._val_batches_saved >= self._max_val_batches:
            return
            
        try:
            import cv2
            from pathlib import Path
            
            save_dir = Path(self.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert images from tensor to numpy
            imgs_np = imgs.cpu().numpy()
            batch_size = imgs_np.shape[0]
            _, c, h, w = imgs_np.shape
            
            # Fetch ground truth from COCO API using image IDs
            targets_list = self._get_gt_from_coco(ids, info_imgs, h, w)
            
            # Create mosaic for labels (ground truth)
            labels_mosaic = self._create_val_mosaic(imgs_np, targets_list, is_prediction=False)
            labels_path = save_dir / f'val_batch{batch_idx}_labels.jpg'
            cv2.imwrite(str(labels_path), labels_mosaic)
            
            # Create mosaic for predictions
            pred_mosaic = self._create_val_mosaic(imgs_np, outputs, is_prediction=True)
            pred_path = save_dir / f'val_batch{batch_idx}_pred.jpg'
            cv2.imwrite(str(pred_path), pred_mosaic)
            
            self._val_batches_saved += 1
            print(f"Saved validation batch {batch_idx} visualization to {save_dir}")
            
        except Exception as e:
            import traceback
            print(f"Warning: Could not save val batch visualization: {e}")
            traceback.print_exc()
    
    def _get_gt_from_coco(self, ids, info_imgs, img_h, img_w):
        """
        Fetch ground truth from COCO API and convert to visualization format.
        
        Args:
            ids: Image IDs (tensor or array, may be shape (N,) or (N, 1))
            info_imgs: Tuple of (heights, widths) for each image in original size
            img_h, img_w: Model input image dimensions (after preprocessing)
            
        Returns:
            List of arrays, each with [x1, y1, x2, y2, conf, cls] in input image coords
        """
        result = []
        
        try:
            coco = self.dataloader.dataset.coco
        except Exception as e:
            print(f"Warning: Could not access COCO API: {e}")
            return [None] * len(ids)
        
        try:
            # info_imgs is (heights_tensor, widths_tensor) - original image sizes
            orig_heights = info_imgs[0].cpu().numpy() if hasattr(info_imgs[0], 'cpu') else np.array(info_imgs[0])
            orig_widths = info_imgs[1].cpu().numpy() if hasattr(info_imgs[1], 'cpu') else np.array(info_imgs[1])
            
            # Handle ids tensor/array - may be shape (N,) or (N, 1)
            if hasattr(ids, 'cpu'):
                ids_np = ids.cpu().numpy()
            else:
                ids_np = np.array(ids)
            
            # Flatten if needed
            ids_np = ids_np.flatten()
            
        except Exception as e:
            print(f"Warning: Could not process input arrays: {e}")
            import traceback
            traceback.print_exc()
            return [None] * len(ids)
        
        for idx in range(len(ids_np)):
            try:
                img_id = int(ids_np[idx])
                ann_ids = coco.getAnnIds(imgIds=img_id)
                annotations = coco.loadAnns(ann_ids)
                
                # Get original image dimensions
                orig_h = float(orig_heights[idx]) if idx < len(orig_heights) else float(img_h)
                orig_w = float(orig_widths[idx]) if idx < len(orig_widths) else float(img_w)
                
                # Compute scale factor (same as preprocessing)
                scale = min(img_h / orig_h, img_w / orig_w)
                
                batch_targets = []
                for ann in annotations:
                    # COCO bbox format: [x, y, width, height]
                    x, y, w, h = ann['bbox']
                    cat_id = ann['category_id']
                    
                    # Map category ID to class index (0-based)
                    cls = cat_id  # Default to category_id
                    if hasattr(self.dataloader.dataset, 'class_ids'):
                        class_ids = self.dataloader.dataset.class_ids
                        if cat_id in class_ids:
                            cls = class_ids.index(cat_id)
                        else:
                            # Try mapping as 0-based index directly if class_ids is sequential
                            if 0 <= cat_id < len(class_ids):
                                cls = cat_id
                            else:
                                continue  # Skip if category not found
                    
                    # Convert to xyxy format and scale to input image size
                    x1 = x * scale
                    y1 = y * scale
                    x2 = (x + w) * scale
                    y2 = (y + h) * scale
                    
                    # Ensure valid box coordinates
                    if x2 > x1 and y2 > y1:
                        # Format: [x1, y1, x2, y2, conf, cls]
                        batch_targets.append([x1, y1, x2, y2, 1.0, cls])
                
                if batch_targets:
                    result.append(np.array(batch_targets))
                else:
                    result.append(None)
                    
            except Exception as e:
                print(f"Warning: Error processing image {idx}: {e}")
                result.append(None)
        
        return result
    
    def _create_val_mosaic(self, imgs_np, boxes_list, is_prediction=True, max_images=16):
        """
        Create a mosaic visualization of validation batch.
        
        Args:
            imgs_np: Numpy array of images (N, C, H, W)
            boxes_list: List of box arrays per image
                       For predictions: [x1, y1, x2, y2, obj_conf, cls_conf, cls] or [x1, y1, x2, y2, conf, cls]
                       For ground truth: [x1, y1, x2, y2, conf, cls] (pixel coords)
            is_prediction: If True, use red color; if False, use green color
            max_images: Maximum number of images to include
            
        Returns:
            Mosaic image as numpy array (H, W, 3)
        """
        import cv2
        import math
        
        batch_size = min(imgs_np.shape[0], max_images)
        
        # Calculate grid size
        grid_size = int(math.ceil(math.sqrt(batch_size)))
        
        # Get image dimensions
        _, c, h, w = imgs_np.shape
        target_size = 640  # Target size for each cell
        
        # Create empty mosaic
        mosaic = np.zeros((grid_size * target_size, grid_size * target_size, 3), dtype=np.uint8)
        
        for idx in range(batch_size):
            # Get image and convert from CHW to HWC
            img = imgs_np[idx].transpose(1, 2, 0)
            
            # Handle different channel formats
            if c == 3:
                img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif c == 1:
                img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
                img = cv2.cvtColor(img.squeeze(), cv2.COLOR_GRAY2BGR)
            else:
                # Multi-channel (e.g., with radar), visualize as event representation
                img = self._event_repr_to_image(img)
            
            # Resize image
            img_resized = cv2.resize(img, (target_size, target_size))
            
            # Scale factor for resized image
            scale_x = target_size / w
            scale_y = target_size / h
            
            # Draw boxes if available
            if boxes_list is not None and idx < len(boxes_list):
                boxes = boxes_list[idx]
                if boxes is not None and len(boxes) > 0:
                    boxes = boxes.cpu().numpy() if hasattr(boxes, 'cpu') else np.array(boxes)
                    
                    for box in boxes:
                        if len(box) < 4:
                            continue
                            
                        # Get coordinates (already in pixel format)
                        x1, y1, x2, y2 = box[:4]
                        
                        # Get confidence and class
                        if is_prediction:
                            # Prediction format: [x1, y1, x2, y2, obj_conf, cls_conf, cls] (7 cols)
                            # or [x1, y1, x2, y2, conf, cls] (6 cols)
                            if len(box) >= 7:
                                conf = float(box[4]) * float(box[5])
                                cls = int(box[6])
                            elif len(box) >= 6:
                                conf = float(box[4])
                                cls = int(box[5])
                            else:
                                conf = float(box[4]) if len(box) > 4 else 1.0
                                cls = 0
                            color = (0, 0, 255)  # Red for predictions
                        else:
                            # Ground truth format: [x1, y1, x2, y2, conf, cls]
                            conf = 1.0
                            cls = int(box[5]) if len(box) > 5 else 0
                            color = (0, 255, 0)  # Green for ground truth
                        
                        # Scale coordinates
                        x1_scaled = int(x1 * scale_x)
                        y1_scaled = int(y1 * scale_y)
                        x2_scaled = int(x2 * scale_x)
                        y2_scaled = int(y2 * scale_y)
                        
                        # Clamp to image bounds
                        x1_scaled = max(0, min(target_size - 1, x1_scaled))
                        y1_scaled = max(0, min(target_size - 1, y1_scaled))
                        x2_scaled = max(0, min(target_size - 1, x2_scaled))
                        y2_scaled = max(0, min(target_size - 1, y2_scaled))
                        
                        # Draw box
                        cv2.rectangle(img_resized, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), color, 2)
                        
                        # Draw label
                        if self.class_names and cls < len(self.class_names):
                            label = f"{self.class_names[cls]}"
                        else:
                            label = f"cls{cls}"
                        if is_prediction:
                            label += f" {conf:.2f}"
                        
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        label_y1 = max(y1_scaled - th - 4, 0)
                        cv2.rectangle(img_resized, (x1_scaled, label_y1), (x1_scaled + tw, y1_scaled), color, -1)
                        cv2.putText(img_resized, label, (x1_scaled, y1_scaled - 2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Place in mosaic
            row = idx // grid_size
            col = idx % grid_size
            y_start = row * target_size
            x_start = col * target_size
            mosaic[y_start:y_start + target_size, x_start:x_start + target_size] = img_resized
        
        return mosaic
    
    def _event_repr_to_image(self, ev_repr):
        """Convert event representation to BGR image for visualization."""
        import cv2
        
        ch = ev_repr.shape[-1]
        ht, wd = ev_repr.shape[:2]
        
        # Handle odd channel counts (e.g., with radar channel)
        if ch > 1 and ch % 2 != 0:
            ev_repr = ev_repr[..., :-1]
            ch = ch - 1
        
        if ch < 2:
            # Single channel - grayscale
            img = ev_repr[..., 0]
            img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Split into positive and negative
        half_ch = ch // 2
        neg_events = ev_repr[..., :half_ch].sum(axis=-1)
        pos_events = ev_repr[..., half_ch:].sum(axis=-1)
        
        # Create visualization - gray background, red for neg, blue for pos
        img = np.ones((ht, wd, 3), dtype=np.uint8) * 127
        
        max_val = max(abs(neg_events).max(), abs(pos_events).max(), 1e-8)
        
        # Positive events -> blue
        pos_mask = pos_events > neg_events
        pos_intensity = np.clip(pos_events / max_val * 255, 0, 255).astype(np.uint8)
        img[pos_mask, 0] = pos_intensity[pos_mask]
        img[pos_mask, 1] = 127
        img[pos_mask, 2] = 127
        
        # Negative events -> red
        neg_mask = neg_events > pos_events
        neg_intensity = np.clip(neg_events / max_val * 255, 0, 255).astype(np.uint8)
        img[neg_mask, 0] = 127
        img[neg_mask, 1] = 127
        img[neg_mask, 2] = neg_intensity[neg_mask]
        
        return img

    def evaluate(
        self, model, distributed=False, half=False, trt_file=None,
        decoder=None, test_size=None, return_outputs=False
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # Reset val batch counter at start of evaluation
        self._val_batches_saved = 0
        
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        output_data = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        for cur_iter, (imgs, targets, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)

                # skip the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                outputs = postprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre
                )
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end
            
            # Save validation batch visualization for first N batches
            if is_main_process() and cur_iter < self._max_val_batches:
                self._save_val_batch_visualization(imgs, outputs, ids, cur_iter, info_imgs)
            
            # Collect confusion matrix data (ground truth vs predictions)
            if is_main_process():
                self._collect_confusion_matrix_data(outputs, ids, info_imgs, imgs.shape[2], imgs.shape[3])

            data_list_elem, image_wise_data = self.convert_to_coco_format(
                outputs, info_imgs, ids, return_outputs=True)
            data_list.extend(data_list_elem)
            output_data.update(image_wise_data)

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            # different process/device might have different speed,
            # to make sure the process will not be stucked, sync func is used here.
            synchronize()
            data_list = gather(data_list, dst=0)
            output_data = gather(output_data, dst=0)
            data_list = list(itertools.chain(*data_list))
            output_data = dict(ChainMap(*output_data))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()

        if return_outputs:
            return eval_results, output_data
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids, return_outputs=False):
        data_list = []
        image_wise_data = defaultdict(dict)
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            image_wise_data.update({
                int(img_id): {
                    "bboxes": [box.numpy().tolist() for box in bboxes],
                    "scores": [score.numpy().item() for score in scores],
                    "categories": [
                        self.dataloader.dataset.class_ids[int(cls[ind])]
                        for ind in range(bboxes.shape[0])
                    ],
                }
            })

            bboxes = xyxy2xywh(bboxes)

            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)

        if return_outputs:
            return data_list, image_wise_data
        return data_list

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            if self.testdev:
                json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
                cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, "w"))
                cocoDt = cocoGt.loadRes(tmp)
            # Try faster evaluators in order of preference
            try:
                from faster_coco_eval import COCOeval_faster as COCOeval
                logger.info("Using faster-coco-eval (fastest).")
            except ImportError:
                try:
                    from yolox.layers import COCOeval_opt as COCOeval
                    logger.info("Using COCOeval_opt (C++ JIT).")
                except Exception:
                    from pycocotools.cocoeval import COCOeval
                    logger.warning("Using standard COCOeval (slow - consider installing faster-coco-eval).")

            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            cat_ids = list(cocoGt.cats.keys())
            cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]
            if self.per_class_AP:
                AP_table = per_class_AP_table(cocoEval, class_names=cat_names)
                info += "per class AP:\n" + AP_table + "\n"
            if self.per_class_AR:
                AR_table = per_class_AR_table(cocoEval, class_names=cat_names)
                info += "per class AR:\n" + AR_table + "\n"
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
