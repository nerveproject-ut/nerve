"""
Distance COCO Evaluator for YOLOX with distance estimation.
Ported from the original deep_old implementation.

This evaluator extends COCOEvaluator to handle distance predictions
and compute distance-specific metrics (Average Distance Error, Distance Invalidity Ratio).
"""
import torch
import time
import itertools
import torchvision
import json
import tempfile
import io
import contextlib
from collections import ChainMap, defaultdict
from tqdm import tqdm
from loguru import logger

from yoloX.yolox.evaluators.coco_evaluator import COCOEvaluator, per_class_AR_table, per_class_AP_table
from custom_cocoeval import Custom_COCOeval

from yoloX.yolox.utils import (
    gather,
    is_main_process,
    synchronize,
    time_synchronized,
    xyxy2xywh
)


class Distance_COCO_Evaluator(COCOEvaluator):
    """
    COCO Evaluator with distance estimation support.
    
    Extends the standard COCOEvaluator to:
    - Handle distance predictions from the model
    - Use Custom_COCOeval for distance-specific metrics
    - Output Average Distance Error (ADE) and Distance Invalidity Ratio (DIR)
    """
    
    def __init__(self, dataloader, img_size: int, confthre: float, nmsthre: float, 
                 num_classes: int, testdev: bool = False, per_class_AP: bool = True, 
                 per_class_AR: bool = True):
        super().__init__(dataloader, img_size, confthre, nmsthre, num_classes, 
                        testdev, per_class_AP, per_class_AR)
        # Store distance metrics for retrieval by visualization module
        self._distance_metrics = {}
    
    def get_distance_metrics(self):
        """Return the computed distance metrics dict."""
        return self._distance_metrics

    def evaluate(
        self, model, distributed=False, half=False, trt_file=None,
        decoder=None, test_size=None, return_outputs=False
    ):
        """
        COCO average precision (AP) Evaluation with distance metrics.
        
        Iterate inference on the test dataset and the results are evaluated by COCO API.
        Also computes distance-specific metrics using Custom_COCOeval.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model: model to evaluate.
            distributed: whether to use distributed evaluation.
            half: whether to use half precision.
            trt_file: TensorRT model file path.
            decoder: optional decoder for outputs.
            test_size: test image size.
            return_outputs: whether to return raw outputs.

        Returns:
            ap50_95 (float): COCO AP of IoU=50:95
            ap50 (float): COCO AP of IoU=50
            summary (str): summary info of evaluation including distance metrics.
        """
        # Reset val batch counter at start of evaluation
        self._val_batches_saved = 0
        
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
            
            # Collect confusion matrix data (for consistency with base class)
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
        """
        Convert model outputs to COCO format with distance predictions.
        
        Args:
            outputs: model outputs including distance predictions
            info_imgs: image info (height, width)
            ids: image ids
            return_outputs: whether to return image-wise data
            
        Returns:
            data_list: list of predictions in COCO format
            image_wise_data: dict of predictions per image (if return_outputs=True)
        """
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

            # Distance predictions (8th column, index 7)
            distances = output[:, 7]

            image_wise_data.update({
                int(img_id): {
                    "bboxes": [box.numpy().tolist() for box in bboxes],
                    "scores": [score.numpy().item() for score in scores],
                    "categories": [
                        self.dataloader.dataset.class_ids[int(cls[ind])]
                        for ind in range(bboxes.shape[0])
                    ],
                    "distances": [d.numpy().item() for d in distances]
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
                    'distance': distances[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)

        if return_outputs:
            return data_list, image_wise_data
        return data_list

    def evaluate_prediction(self, data_dict, statistics):
        """
        Evaluate predictions using Custom_COCOeval for distance metrics.
        
        Args:
            data_dict: list of predictions in COCO format
            statistics: inference statistics [inference_time, nms_time, n_samples]
            
        Returns:
            ap50_95: COCO AP at IoU=0.50:0.95
            ap50: COCO AP at IoU=0.50
            info: string with evaluation results including distance metrics
        """
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

            # Use Custom_COCOeval for distance metrics
            cocoEval = Custom_COCOeval(cocoGt, cocoDt, annType[1])
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
            
            # Add distance estimation metrics summary (similar to ReYOLOv8 format)
            info += self._format_distance_metrics(cocoEval, data_dict)
            
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            # No detections - provide informative message
            info += "\n" + "=" * 70 + "\n"
            info += "Distance Estimation Metrics\n"
            info += "=" * 70 + "\n"
            info += "  No detections produced. Distance metrics cannot be computed.\n"
            info += "  This may be due to:\n"
            info += "    - Model undertrained (too few epochs)\n"
            info += "    - High confidence threshold\n"
            info += "    - Model not learning detection task\n"
            info += "=" * 70 + "\n"
            return 0, 0, info
    
    def _format_distance_metrics(self, cocoEval, data_dict):
        """
        Format distance estimation metrics using COCO-style matching.
        
        This method extracts distance metrics from properly matched detection-GT pairs
        using the COCO evaluation's gtMatches, ensuring consistency with detection metrics.
        
        Args:
            cocoEval: Custom_COCOeval instance with evaluation results
            data_dict: List of predictions in COCO format
            
        Returns:
            info: Formatted string with distance metrics
        """
        import numpy as np
        
        info = "\n" + "=" * 70 + "\n"
        info += "Distance Estimation Metrics (COCO-style matching)\n"
        info += "=" * 70 + "\n"
        
        # Extract distance metrics from Custom_COCOeval results
        # These are computed using proper COCO matching (gtMatches)
        if hasattr(cocoEval, 'eval') and 'dist_error' in cocoEval.eval:
            dist_error = cocoEval.eval['dist_error']
            dist_invalid = cocoEval.eval['dist_invalid_ratio']
            
            # Extract valid values (non -1)
            valid_errors = dist_error[dist_error > -1]
            valid_invalid = dist_invalid[dist_invalid > -1]
            
            # Primary metric: ADE from COCO matching (most reliable)
            if len(valid_errors) > 0:
                ade = np.mean(valid_errors)
                ade_std = np.std(valid_errors)
                info += f"  ADE (COCO matching):     {ade:.3f} m (std: {ade_std:.3f})\n"
                
                # Store for visualization
                self._distance_metrics['ade_coco'] = ade
                self._distance_metrics['ade_coco_std'] = ade_std
            
            if len(valid_invalid) > 0:
                dir_ratio = np.mean(valid_invalid) * 100
                info += f"  Invalid Distance Ratio:  {dir_ratio:.1f}%\n"
                self._distance_metrics['invalid_ratio'] = dir_ratio
        
        # Additionally compute detailed metrics from matched pairs
        # Use IoU-based matching instead of "first GT" matching
        cocoGt = self.dataloader.dataset.coco
        cocoDt = cocoEval.cocoDt
        
        errors = []
        pred_distances = []
        gt_distances = []
        
        # Build prediction lookup by ID for fast access
        dt_by_id = {ann['id']: ann for ann in cocoDt.anns.values()}
        
        # Use COCO's evaluation results to get properly matched pairs
        # evalImgs contains matching information per image
        if hasattr(cocoEval, 'evalImgs') and cocoEval.evalImgs:
            for evalImg in cocoEval.evalImgs:
                if evalImg is None:
                    continue
                
                # gtMatches[iou_idx, det_idx] contains matched GT id (0 if unmatched)
                # Use IoU threshold index 0 (typically 0.5)
                gt_matches = evalImg.get('gtMatches', None)
                gt_ids = evalImg.get('gtIds', [])
                dt_ids = evalImg.get('dtIds', [])
                
                if gt_matches is None or len(gt_ids) == 0:
                    continue
                
                # For each matched GT-DT pair at IoU threshold 0 (0.5)
                for gt_idx, gt_id in enumerate(gt_ids):
                    if gt_idx >= gt_matches.shape[1]:
                        continue
                    
                    # Get matched detection ID at IoU threshold 0
                    dt_id = int(gt_matches[0, gt_idx]) if gt_matches.shape[0] > 0 else 0
                    
                    if dt_id == 0:
                        continue  # No match for this GT
                    
                    # Get GT and DT annotations
                    gt_ann = cocoGt.anns.get(gt_id)
                    dt_ann = dt_by_id.get(dt_id)
                    
                    if gt_ann is None or dt_ann is None:
                        continue
                    
                    gt_dist = gt_ann.get('avg_distance', gt_ann.get('distance', -1))
                    pred_dist = dt_ann.get('distance', -1)
                    
                    # Only compute error for valid distances
                    if gt_dist >= 0 and pred_dist >= 0:
                        error = abs(pred_dist - gt_dist)
                        errors.append(error)
                        pred_distances.append(pred_dist)
                        gt_distances.append(gt_dist)
        
        # Compute detailed metrics from matched pairs
        if len(errors) > 0:
            errors = np.array(errors)
            pred_distances = np.array(pred_distances)
            gt_distances = np.array(gt_distances)
            
            mae = np.mean(errors)
            rmse = np.sqrt(np.mean(np.square(pred_distances - gt_distances)))
            median_ae = np.median(errors)
            max_error = np.max(errors)
            min_error = np.min(errors)
            
            # Accuracy at thresholds
            acc_05 = np.mean(errors <= 0.5) * 100
            acc_10 = np.mean(errors <= 1.0) * 100
            acc_20 = np.mean(errors <= 2.0) * 100
            
            # Store metrics for visualization module
            self._distance_metrics.update({
                'samples': len(errors),
                'mae': mae,
                'rmse': rmse,
                'median_ae': median_ae,
                'max_error': max_error,
                'min_error': min_error,
                'acc_05': acc_05,
                'acc_10': acc_10,
                'acc_20': acc_20,
            })
            
            info += f"\n  Detailed Metrics (IoU@0.5 matched pairs):\n"
            info += f"  ─────────────────────────────────────────\n"
            info += f"  Matched Samples: {len(errors)}\n"
            info += f"  MAE:             {mae:.3f} m\n"
            info += f"  RMSE:            {rmse:.3f} m\n"
            info += f"  Median AE:       {median_ae:.3f} m\n"
            info += f"  Min Error:       {min_error:.3f} m\n"
            info += f"  Max Error:       {max_error:.3f} m\n"
            info += f"  Acc @ 0.5m:      {acc_05:.1f}%\n"
            info += f"  Acc @ 1.0m:      {acc_10:.1f}%\n"
            info += f"  Acc @ 2.0m:      {acc_20:.1f}%\n"
        else:
            info += "  No valid matched pairs with distance data.\n"
        
        info += "=" * 70 + "\n"
        return info


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    """
    Postprocess model predictions with distance handling.
    
    Performs:
    - Box coordinate conversion (center to corner format)
    - Confidence thresholding
    - Non-maximum suppression
    - Distance prediction extraction
    
    Args:
        prediction: raw model predictions
        num_classes: number of classes
        conf_thre: confidence threshold
        nms_thre: NMS IoU threshold
        class_agnostic: whether to perform class-agnostic NMS
        
    Returns:
        output: list of detections per image, each with format:
                [x1, y1, x2, y2, obj_conf, class_conf, class_pred, distance]
    """
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        # Distance prediction is at index 6 (after class predictions)
        dist_pred = image_pred[:, 6].unsqueeze(-1)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred, distance)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float(), dist_pred), 1)

        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output
