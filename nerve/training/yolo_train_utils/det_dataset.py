import numpy as np
import os
import torch
import cv2
from torch.utils.data import Dataset
from nerve.training.yolo_train_utils.config import cfg
from nerve.training.yolo_train_utils.utils import bbox_iou


class FusionDetDataset(Dataset):
    """ Dataset for XRay Det application (one image no sequence) """

    def __init__(self, anchors, strides, train=True):
        """
        :param anchors: information for anchors
        :param strides: list of strides
        :param train: if true use training mode, else use validation mode
        """
        self.annot_path = cfg['TRAIN']['ANNOT_PATH'] if train else cfg['VAL']['ANNOT_PATH']
        self.input_sizes = cfg['TRAIN']['INPUT_SIZE']
        self.batch_size = cfg['TRAIN']['BATCH_SIZE'] if train else cfg['VAL']['BATCH_SIZE']

        self.train_input_size = cfg['TRAIN']['INPUT_SIZE']
        self.strides = np.array(strides)
        self.num_scale = len(self.strides)
        self.train_output_size_h = np.int0(self.train_input_size[0] / self.strides)
        self.train_output_size_w = np.int0(self.train_input_size[1] / self.strides)

        self.anchors = anchors
        self.anchor_per_scale = cfg['YOLO']['ANCHOR_PER_SCALE']
        self.max_bbox_per_scale = 150

        self.annotations = self.load_annotations()
        self.num_samples = len(self.annotations)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        """
        Get item base on index

        :param item: index of item
        :return: image, label for each scale, bbox for each scale
        """
        annotation = self.annotations[item]
        image, bboxes = self.parse_annotation(annotation)
        label_list, bbox_list = self.preprocess_true_boxes(bboxes)
        # print(image.shape) # (128, 256, 2)
        # print(image[:10, :10, 0]) # grey scale cam image
        # print(image[:10, :10, 1]) # sparse point cloud
        image = torch.Tensor(image).permute(2, 0, 1)
        label_list = [torch.Tensor(label) for label in label_list]
        bbox_list = [torch.Tensor(bbox) for bbox in bbox_list]
        return image, *label_list, *bbox_list

    def load_annotations(self):
        """
        Load all annotations from dataset.txt file

        :return: annotations
        """
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split("*")[1:]) != 0]
        return annotations

    def parse_annotation(self, annotation):
        """
        Parse annotation from dataset.txt

        :param annotation: single string annotation
        :return: image, bboxes
        """
        line = annotation.split("*")
        image_path = line[0]
        pc_path = line[1]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " % image_path)
        if not os.path.exists(pc_path):
            raise KeyError("%s does not exist ... " % pc_path)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        point_cloud = np.load(pc_path)
        new_image = np.stack([image, point_cloud], axis=2)
        # bbox - 4 number for box + 1 number for distance
        # Here bbox is encoded by (x_min, y_min, x_max, y_max, distance)
        bboxes = []
        for box in line[2:]:
            single_box = box.split(',')
            single_box = [int(single_box[0]), int(single_box[1]), int(single_box[2]), int(single_box[3]), float(single_box[4])]
            bboxes.append(single_box)
        bboxes = np.array(bboxes)

        return new_image, bboxes

    def preprocess_true_boxes(self, bboxes):
        """
        Preprocess ground truth bboxes and put positive sample to corresponding grid cell and anchor position

        :param bboxes: list of bboxes
        :return: label, bboxes_xywh
        """

        label = [np.zeros((self.train_output_size_h[i], self.train_output_size_w[i], self.anchor_per_scale, 6)) for i in range(self.num_scale)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(self.num_scale)]
        bbox_count = np.zeros((self.num_scale,))

        for bbox in bboxes:
            # bbox coordinate and distance
            bbox_coor = bbox[:4]
            bbox_distance = bbox[4]

            # transform bbox coordinate to center of bbox (x, y), width and height of bbox (w, h)
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            # scale bbox info to each different scale
            # This is only used to compute similar anchor and map bbox location to each scale
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(self.num_scale):
                # Put anchor box to the position of the bbox
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                # Compare real bbox and 3 anchor box and create a mask with a threshold
                iou_scale = bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    # Put label for anchor over threshold with positive bbox and label
                    # If there are multiple box for the same anchor in same grid cell, this can be overlaied
                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:6] = bbox_distance

                    # Collect all bbox for one scale
                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            # If there is no anchor in all scales over threshold, pick the best one as positive
            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:6] = bbox_distance

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        return label, bboxes_xywh
