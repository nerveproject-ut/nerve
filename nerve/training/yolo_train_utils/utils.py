import torch
import random
import cv2
import numpy as np
from nerve.training.yolo_train_utils.config import cfg


def get_anchors(anchors_path, device):
    """
    Load anchors from a file and transform to torch tensor

    :param anchors_path: path to anchor file
    :param device: torch device
    :return: anchors, anchors_tensor
    """
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    anchors = anchors.reshape(-1, cfg['YOLO']['ANCHOR_PER_SCALE'], 2)
    anchors_tensor = torch.Tensor(anchors).to(device)
    return anchors, anchors_tensor


def image_preprocess(image, target_size, gt_boxes=None):
    """
    Preprocess image and ground truth bbox to target size

    :param image: image
    :param target_size: target size
    :param gt_boxes: ground truth bbox
    :return: image_padded, gt_boxes
    """
    ih, iw = target_size
    h, w = image.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 1], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh:nh + dh, dw:nw + dw, 0] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


def bbox_iou(boxes1, boxes2):
    """
    Compute IOU between two group of bboxes (using numpy)

    :param boxes1: bbox group 1
    :param boxes2: bbox group 2
    :return: iou
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                             boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                             boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return inter_area / np.maximum(union_area, 1e-12)
