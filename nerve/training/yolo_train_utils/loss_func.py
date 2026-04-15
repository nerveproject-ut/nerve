import torch
import torch.nn.functional as F
import numpy as np
from nerve.training.yolo_train_utils.config import cfg


def bbox_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """
    Compute IOU between two group of bboxes

    :param boxes1: bbox group 1
    :param boxes2: bbox group 2
    :return: iou
    """
    # Transform xywh representation of bbox to (x_min, y_min, x_max, y_max) representation
    boxes1 = torch.cat((boxes1[..., :2] - boxes1[..., 2:] * 0.5, boxes1[..., :2] + boxes1[..., 2:] * 0.5), dim=-1)
    boxes2 = torch.cat((boxes2[..., :2] - boxes2[..., 2:] * 0.5, boxes2[..., :2] + boxes2[..., 2:] * 0.5), dim=-1)
    boxes1 = torch.cat((torch.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        torch.maximum(boxes1[..., :2], boxes1[..., 2:])), dim=-1)
    boxes2 = torch.cat((torch.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        torch.maximum(boxes2[..., :2], boxes2[..., 2:])), dim=-1)

    # Compute area for boxes 1 and boxes 2
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # Find left up point and right down point for the intersection area of box 1 and box 2
    left_up = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])

    # Compute IOU of two boxes
    inter_section = torch.clamp_min(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / torch.clamp_min(union_area, 1e-12)

    return iou


def bbox_giou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """
    Compute GIOU between two group of bboxes

    :param boxes1: bbox group 1
    :param boxes2: bbox group 2
    :return: giou
    """
    # Transform xywh representation of bbox to (x_min, y_min, x_max, y_max) representation
    boxes1 = torch.cat((boxes1[..., :2] - boxes1[..., 2:] * 0.5, boxes1[..., :2] + boxes1[..., 2:] * 0.5), dim=-1)
    boxes2 = torch.cat((boxes2[..., :2] - boxes2[..., 2:] * 0.5, boxes2[..., :2] + boxes2[..., 2:] * 0.5), dim=-1)
    boxes1 = torch.cat((torch.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        torch.maximum(boxes1[..., :2], boxes1[..., 2:])), dim=-1)
    boxes2 = torch.cat((torch.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        torch.maximum(boxes2[..., :2], boxes2[..., 2:])), dim=-1)

    # Compute area for boxes 1 and boxes 2
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # Find left up point and right down point for the intersection area of box 1 and box 2
    left_up = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])

    # Compute IOU of two boxes
    inter_section = torch.clamp_min(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / torch.clamp_min(union_area, 1e-12)

    # Compute GIOU by computing the smallest enclosing convex object between box 1 and box 2
    enclose_left_up = torch.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = torch.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = torch.clamp_min(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / torch.clamp_min(enclose_area, 1e-12)

    return giou


def compute_loss(pred: torch.Tensor, conv: torch.Tensor, label: torch.Tensor, bboxes: torch.Tensor, strides, i=0):
    """
    Compute YOLO training loss

    YOLO training loss is consisted of 3 loss items: giou_loss, conf_loss, prob_loss

    :param pred: Network prediction after decode
    :param conv: Network raw convolution output
    :param label: ground truth positive samples with label and bbox info
    :param bboxes: list of ground truth bbox information
    :param strides: list of strides
    :param i: scale idx
    :return: giou_loss, conf_loss, prob_loss
    """
    conv_shape = conv.size()
    batch_size = conv_shape[0]
    output_size_h = conv_shape[1]
    output_size_w = conv_shape[2]
    input_size_h = float(strides[i] * output_size_h)
    input_size_w = float(strides[i] * output_size_w)

    # Get information from raw convolution output, decoded prediction, and ground truth
    conv_raw_conf = conv[:, :, :, :, 4:5]

    pred_xywh = pred[:, :, :, :, 0:4]
    pred_conf = pred[:, :, :, :, 4:5]
    pred_dist = pred[:, :, :, :, 5:6]

    label_xywh = label[:, :, :, :, 0:4]
    respond_bbox = label[:, :, :, :, 4:5]
    label_dist = label[:, :, :, :, 5:6]

    # Compute GIOU loss for bbox position and size regression
    giou = bbox_giou(pred_xywh, label_xywh).unsqueeze(-1)
    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size_h * input_size_w)
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

    # Compute Confidence loss for foreground and background
    iou = bbox_iou(pred_xywh.view(batch_size, output_size_h, output_size_w, cfg['YOLO']['ANCHOR_PER_SCALE'], 1, 4),
                   bboxes.view(batch_size, 1, 1, 1, -1, 4))
    max_iou = torch.max(iou, dim=-1, keepdim=True).values
    respond_bgd = (1.0 - respond_bbox) * (max_iou < cfg['YOLO']['IOU_LOSS_THRESHOLD']).float()
    conf_focal = torch.pow(respond_bbox - pred_conf, 2)
    conf_loss = conf_focal * (
        respond_bbox * F.binary_cross_entropy_with_logits(input=conv_raw_conf, target=respond_bbox, reduction='none')
        +
        respond_bgd * F.binary_cross_entropy_with_logits(input=conv_raw_conf, target=respond_bbox, reduction='none')
    )

    # Compute distance MSE loss for positive samples
    rescale_pred_dist = pred_dist * (cfg['YOLO']['MAX_DIST'] - cfg['YOLO']['MIN_DIST']) + cfg['YOLO']['MIN_DIST']
    dist_loss = F.mse_loss(input=rescale_pred_dist, target=label_dist, reduction='none')

    # Sum loss across all dimensions and compute mean across batches
    giou_loss = torch.mean(torch.sum(giou_loss, dim=(1, 2, 3, 4)))
    conf_loss = torch.mean(torch.sum(conf_loss, dim=(1, 2, 3, 4)))
    dist_loss = torch.mean(torch.sum(dist_loss, dim=(1, 2, 3, 4)))

    return giou_loss, conf_loss, dist_loss
