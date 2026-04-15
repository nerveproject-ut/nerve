import torch
from nerve.training.yolo_train_utils.config import cfg


def decode(conv_output: torch.Tensor, device, anchors, strides, i=0):
    """
    Decode raw convolution output to bbox information

    return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + 1]
            contains (x, y, w, h, score, distance)

    :param conv_output: raw convolution output from network
    :param device: device for computation
    :param anchors: information for anchors
    :param strides: list of strides
    :param i: scale idx
    :return: conv_output, conv_decode
    """
    conv_shape = conv_output.size()
    batch_size = conv_shape[0]
    output_size_h = conv_shape[2]
    output_size_w = conv_shape[3]

    # Get information from raw convolution output
    conv_output = conv_output.permute(0, 2, 3, 1)
    conv_output = conv_output.view(batch_size, output_size_h, output_size_w, cfg['YOLO']['ANCHOR_PER_SCALE'], 6)
    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_dist = conv_output[:, :, :, :, 5:6]

    # Generate (x, y) index for each grid cell
    y = torch.tile(torch.arange(output_size_h, device=device).view(-1, 1), [1, output_size_w])
    x = torch.tile(torch.arange(output_size_w, device=device).view(1, -1), [output_size_h, 1])

    xy_grid = torch.cat((x.view(output_size_h, output_size_w, 1),
                         y.view(output_size_h, output_size_w, 1)), dim=-1)
    xy_grid = torch.tile(xy_grid.view(1, output_size_h, output_size_w, 1, 2),
                         [batch_size, 1, 1, cfg['YOLO']['ANCHOR_PER_SCALE'], 1]).float()

    # Shift sigmoid(dx, dy) on grid cell index, then multiply stride to recover the value to input image dimension
    pred_xy = (torch.sigmoid(conv_raw_dxdy) + xy_grid) * strides[i]
    # Compute width and height by doing exp(dw, dh) and multiply anchor
    # then multiply stride to recover the value to input image dimension
    pred_wh = (torch.exp(conv_raw_dwdh) * anchors[i]) * strides[i]

    pred_xywh = torch.cat((pred_xy, pred_wh), dim=-1)
    pred_conf = torch.sigmoid(conv_raw_conf)
    pred_dist = torch.sigmoid(conv_raw_dist)
    conv_decode = torch.cat((pred_xywh, pred_conf, pred_dist), dim=-1)

    return conv_output, conv_decode
