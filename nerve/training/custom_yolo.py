from yoloX.yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

from yoloX.yolox.utils import bboxes_iou, cxcywh2xyxy, meshgrid, xyxy2xywh

import random

from yoloX.yolox.models.losses import IOUloss
from yoloX.yolox.models.network_blocks import BaseConv, DWConv


def mask_image(img:torch.Tensor, xmin:torch.Tensor, xmax:torch.Tensor, ymin:torch.Tensor, ymax:torch.Tensor):

    h, w = img.shape
    d = len(xmin)
    
    images = img.unsqueeze(1).repeat(1, d, 1)
    xmin = xmin.unsqueeze(-1)
    xmax = xmax.unsqueeze(-1)
    ymin = ymin.unsqueeze(0)
    ymax = ymax.unsqueeze(0)

    w_range = torch.arange(w, device=img.device).unsqueeze(0)
    h_range = torch.arange(h, device=img.device).unsqueeze(-1)

    images[h_range < ymin, :] = 0
    images[h_range >= ymax, :] = 0
    images[:, w_range < xmin] = 0
    images[:, w_range >= xmax] = 0

    images = torch.moveaxis(images, 1, 0)

    return images

class Customized_YOLOX(YOLOX):
    def __init__(self, backbone=None, head=None, distance_from_head=True, max_radar_distance=10.0, original_img_size=(640, 360)):
        super().__init__(backbone, head)
        self.distance_from_head = distance_from_head
        self.original_img_size = original_img_size      # (w,h)
        self.max_radar_dist = max_radar_distance

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            if self.distance_from_head:
                loss, iou_loss, conf_loss, cls_loss, l1_loss, loss_distance, num_fg = self.head(
                    fpn_outs, targets, x
                )
                outputs = {
                    "total_loss": loss,
                    "iou_loss": iou_loss,
                    "l1_loss": l1_loss,
                    "conf_loss": conf_loss,
                    "cls_loss": cls_loss,
                    "dist_loss": loss_distance,
                    "num_fg": num_fg,
                }
            else:
                loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                    fpn_outs, targets, x
                )
                outputs = {
                    "total_loss": loss,
                    "iou_loss": iou_loss,
                    "l1_loss": l1_loss,
                    "conf_loss": conf_loss,
                    "cls_loss": cls_loss,
                    "num_fg": num_fg,
                }
        else:
            outputs = self.head(fpn_outs)

            #to be finished!
            
            
            if not self.distance_from_head:
                # then we need to infer distance by studing the radar map, using generated BB

                # input data (x) has shape:         (b, c, h, w)
                # output data (outputs) has shape:  (b, d, 6)       where d is the (max) number of detections

                batch_size, _, h, w = x.shape
                ratio = max(self.original_img_size[0]/w, self.original_img_size[1]/h)
                resized_h, resized_w = round(self.original_img_size[1] / ratio), round(self.original_img_size[0] / ratio)
                

                # radar maps are encoded in the third channel of input data
                dist_maps_from_radar = x[:, 2].to(torch.uint8)

                xmin = torch.round(outputs[:, :, 0] - outputs[:, :, 2] / 2).clamp(0, resized_w)
                ymin = torch.round(outputs[:, :, 1] - outputs[:, :, 3] / 2).clamp(0, resized_h)
                xmax = torch.round(outputs[:, :, 0] + outputs[:, :, 2] / 2).clamp(0, resized_w)
                ymax = torch.round(outputs[:, :, 1] + outputs[:, :, 3] / 2).clamp(0, resized_h)

                num_bbs = xmin.shape[1]
                dist = torch.empty((batch_size,  num_bbs, 1), device=outputs.device)
                outputs = torch.cat((outputs, dist), -1)
                chunks = 1
                for b in range(batch_size):
                    
                    #NOTE:  To be sure to have data small enough to fit in memory, here it follows a simple mechanism
                    #       which subvide the workload in smaller steps whenever we don't have enough memory. 
                    while True:
                        try:
                            for i in range(chunks):
                                lower_idx = int(num_bbs * i / chunks)
                                upper_idx = int(num_bbs * (i+1) / chunks)
                                cropped_dist_maps = mask_image(dist_maps_from_radar[b],
                                                            xmin[b][lower_idx:upper_idx], xmax[b][lower_idx:upper_idx],
                                                            ymin[b][lower_idx:upper_idx], ymax[b][lower_idx:upper_idx])
                                
                                #let's calculate the average distance using radar pixels (valid only where different from 0)
                                nonzeros_count = torch.count_nonzero(cropped_dist_maps, (1, 2))
                                mask_nonzeros = nonzeros_count != 0
                                distances = cropped_dist_maps.to(torch.float).sum((1, 2))
                                distances[nonzeros_count == 0] = -1.0
                                distances[mask_nonzeros] *= (self.max_radar_dist / 255.0)
                                distances[mask_nonzeros] /= nonzeros_count[mask_nonzeros]

                                #let's store the results. NOTE that -1 is assigned where we have no point-cloud points.
                                outputs[b, lower_idx:upper_idx, -1] = distances
                            break
                        except torch.cuda.OutOfMemoryError:
                            chunks+=1
                            print(f"To fit in memory, subdiving workload in {chunks} chunks.")
                            continue

            
                # debug and visualization.
                """max_conf_idx = torch.argmax(outputs[0, :, 4])
                conf_val = round(float(outputs[0, max_conf_idx, 4])*100)
                dist_val = float(outputs[0, max_conf_idx, -1])
                print(f'Max confidence in {max_conf_idx} : {conf_val}%. Distance: {dist_val :.3f} meters.')
                pt1 = (int(xmin[0, max_conf_idx]), int(ymin[0, max_conf_idx]))
                pt2 = (int(xmax[0, max_conf_idx]), int(ymax[0, max_conf_idx]))
                
                for i in range(3):
                    img = x[0, i].cpu().numpy() #* 255
                    img = img.astype(np.uint8)

                    
                    cv2.rectangle(img, pt1, pt2, 255, 1)
                    cv2.imshow(f"frame_{i}", img)
                cv2.waitKey(0)"""


            el = outputs[:, :, 0]
            outputs = torch.cat((outputs, torch.zeros_like(el).unsqueeze(-1)),dim=-1)

        return outputs


class YOLOX_custom_distance_head(YOLOXHead):
    def __init__(self, num_classes, width=1, strides=[8, 16, 32], in_channels=[256, 512, 1024], act="silu", depthwise=False, min_dist=0.0, max_dist=10.0, nbins=100):
        super().__init__(num_classes, width, strides, in_channels, act, depthwise)
        
        # adding a new head to the regression branch.
        self.dist_convs = nn.ModuleList()                                               #diff
        self.dist_preds = nn.ModuleList()                                               #diff

        Conv = DWConv if depthwise else BaseConv

        #diff
        self.dist_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )


        #diff
        for i in range(len(in_channels)):
            self.dist_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=nbins,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
        
        #diff
        self.min_distance = min_dist
        self.max_distance = max_dist
        self.nbins = nbins
        self.distance_loss_multiplier = 1.0

        self.dist_multiplier = (max_dist - min_dist) / nbins

    
    def initialize_biases(self, prior_prob):
        super().initialize_biases(prior_prob)

        #diff
        for conv in self.dist_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        #diff
        for k, (cls_conv, reg_conv, dist_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.dist_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x
            dist_x = x              #diff

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            dist_feat = dist_conv(dist_x)               #diff
            dist_output = self.dist_preds[k](dist_feat) #diff

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output, dist_output], 1) #diff
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, 1, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())

            else:
                distances = self.evaluate_distances(dist_output, self.dist_multiplier) #diff
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid(), distances], 1 #diff
                )

            outputs.append(output)

        if self.training:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs


    #diff
    def evaluate_distances(self, dist_output:torch.Tensor, dist_multiplier:float, use_only_max=True):
        if use_only_max:
            distances = torch.argmax(dist_output.sigmoid(), dim=1).unsqueeze(1) * dist_multiplier
        else:
            dist_sigm = dist_output.sigmoid()
            dist_sigm /= torch.sum(dist_sigm, 1).unsqueeze(1)
            distances = torch.sum(dist_sigm * dist_multiplier, 1)
            distances = distances.unsqueeze(1)
            
        return distances

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.nbins + self.num_classes                                         #diff
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, 1, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid


    def get_losses(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
    ):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5].unsqueeze(-1)  # [batch, n_anchors_all, n_cls]
        dist_preds = outputs[:, :, 6:].unsqueeze(-1)                                     #diff

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        dist_targets = []                                                   #diff
        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                dist_target = outputs.new_zeros((0, self.nbins))            #diff
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                
                gt_distances = labels[batch_idx, :num_gt, 5] #diff
            
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        obj_preds,
                    )
                except RuntimeError as e:
                    # TODO: the string might change, consider a better way
                    if "CUDA out of memory. " not in str(e):
                        raise  # RuntimeError might not caused by CUDA OOM

                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        obj_preds,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]

                #diff
                #let's convert these distances (scalars) in the classification-like vector
                gt_distances_clamped = torch.clamp(gt_distances[matched_gt_inds], self.min_distance, self.max_distance)
                n_el = len(gt_distances_clamped)
                dist_target = gt_distances.new_zeros((n_el, self.nbins)).flatten()
                if n_el > 0:
                    idx = (gt_distances_clamped / self.dist_multiplier).to(torch.int)
                    idx = torch.clamp(idx, 0, self.nbins-1)
                    idx += torch.arange(0, n_el).to(gt_distances.device) * self.nbins
                    dist_target[idx.long()] = 1
                    dist_target = dist_target.view(n_el, self.nbins)
                


                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            dist_targets.append(dist_target)                                #diff
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        dist_targets = torch.cat(dist_targets, 0)                           #diff
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg
        reg_weight = 5.0
        loss_iou *= reg_weight

        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg

        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg

        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0


        #diff
        loss_distance = 0
        loss_distance = (
            self.bcewithlog_loss(dist_preds.view(-1, self.nbins)[fg_masks], dist_targets)
        ).sum() / num_fg


        

        
        loss_distance = self.distance_loss_multiplier * loss_distance                   #diff
        loss = loss_iou + loss_obj + loss_cls + loss_l1 + loss_distance    #diff

        return (
            loss,
            loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            loss_distance,                                                  #diff
            num_fg / max(num_gts, 1),
        )

    def visualize_assign_result(self, xin, labels=None, imgs=None, save_prefix="assign_vis_"):
        # original forward logic
        outputs, x_shifts, y_shifts, expanded_strides = [], [], [], []
        # TODO: use forward logic here.

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)
            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)
            dist_output = self.dist_preds[k](reg_feat)                                  #diff

            output = torch.cat([reg_output, obj_output, cls_output, dist_output], 1)    #diff
            output, grid = self.get_output_and_grid(output, k, stride_this_level, xin[0].type())
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(
                torch.full((1, grid.shape[1]), stride_this_level).type_as(xin[0])
            )
            outputs.append(output)

        outputs = torch.cat(outputs, 1)
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5].unsqueeze(-1)  # [batch, n_anchors_all, n_cls]     #diff
        dist_preds = outputs[:, :, 6].unsqueeze(-1)                                     #diff

        # calculate targets
        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)

        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects
        for batch_idx, (img, num_gt, label) in enumerate(zip(imgs, nlabel, labels)):
            img = imgs[batch_idx].permute(1, 2, 0).to(torch.uint8)
            num_gt = int(num_gt)
            if num_gt == 0:
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = label[:num_gt, 1:5]
                gt_classes = label[:num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]
                _, fg_mask, _, matched_gt_inds, _ = self.get_assignments(  # noqa
                    batch_idx, num_gt, gt_bboxes_per_image, gt_classes,
                    bboxes_preds_per_image, expanded_strides, x_shifts,
                    y_shifts, cls_preds, obj_preds,
                )

            img = img.cpu().numpy().copy()  # copy is crucial here
            coords = torch.stack([
                ((x_shifts + 0.5) * expanded_strides).flatten()[fg_mask],
                ((y_shifts + 0.5) * expanded_strides).flatten()[fg_mask],
            ], 1)

            xyxy_boxes = cxcywh2xyxy(gt_bboxes_per_image)
            save_name = save_prefix + str(batch_idx) + ".png"
            
            img = visualize_assign(img, xyxy_boxes, coords, matched_gt_inds, save_name)
            logger.info(f"save img to {save_name}")


def random_color():
    return random.randint(0, 128), random.randint(0, 255), random.randint(0, 255)

def visualize_assign(img:np.ndarray, boxes, coords, match_results, save_name=None, saturate_img=True) -> np.ndarray:
    """visualize label assign result.

    Args:
        img: img to visualize
        boxes: gt boxes in xyxy format
        coords: coords of matched anchors
        match_results: match results of each gt box and coord.
        save_name: name of save image, if None, image will not be saved. Default: None.
    """
    if saturate_img:
        img = (img * 255).clip(0, 255)

    for box_id, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        color = random_color()
        if len(coords) > 0:
            assign_coords = coords[match_results == box_id]
        else:
            assign_coords = coords
        if assign_coords.numel() == 0:
            # unmatched boxes are red
            color = (0, 0, 255)
            cv2.putText(
                img, "unmatched", (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1
            )
        else:
            for coord in assign_coords:
                # draw assigned anchor
                cv2.circle(img, (int(coord[0]), int(coord[1])), 3, color, -1)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    if save_name is not None:
        cv2.imwrite(save_name, img)

    return img