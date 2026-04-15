#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   simple_extractor.py
@Time    :   8/30/19 8:59 PM
@Desc    :   Simple Extractor
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import nerve.extraction.segmentation.networks as networks
from nerve.extraction.segmentation.utils.transforms import transform_logits
from nerve.extraction.segmentation.datasets.simple_extractor_dataset import SimpleFolderDataset

dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    },
    'atr': {
        'input_size': [512, 512],
        'num_classes': 18,
        'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                  'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
    },
    'pascal': {
        'input_size': [512, 512],
        'num_classes': 7,
        'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
    }
}




def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Self Correction for Human Parsing")

    parser.add_argument("--dataset", type=str, default='lip',
                        choices=['lip', 'atr', 'pascal'])
    parser.add_argument("--model-restore", type=str, default='',
                        help="restore pretrained model parameters.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--input-dir", type=str, default='',
                        help="path of input image folder.")
    parser.add_argument("--output-dir", type=str, default='',
                        help="path of output image folder.")
    parser.add_argument("--logits", action='store_true',
                        default=False, help="whether to save the logits.")

    return parser.parse_args()


def main():
    args = get_arguments()

    gpus = [int(i) for i in args.gpu.split(',')]
    assert len(gpus) == 1
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    num_classes = dataset_settings[args.dataset]['num_classes']
    input_size = dataset_settings[args.dataset]['input_size']
    label = dataset_settings[args.dataset]['label']
    print("Evaluating total class number {} with {}".format(num_classes, label))

    model = networks.init_model(
        'resnet101', num_classes=num_classes, pretrained=None)

    state_dict = torch.load(args.model_restore)['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[
                             0.225, 0.224, 0.229])
    ])
    dataset = SimpleFolderDataset(
        root=args.input_dir, input_size=input_size, transform=transform)
    dataloader = DataLoader(dataset)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader)):
            image, meta = batch
            img_name = meta['name'][0]
            c = meta['center'].numpy()[0]
            s = meta['scale'].numpy()[0]
            w = meta['width'].numpy()[0]
            h = meta['height'].numpy()[0]

            print(image.shape)
            print(c,s,w,h)

            output = model(image.cuda())
            upsample = torch.nn.Upsample(
                size=input_size, mode='bilinear', align_corners=True)
            upsample_output = upsample(output[0][-1][0].unsqueeze(0))
            upsample_output = upsample_output.squeeze()
            upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

            print("output shape: ", upsample_output.data.cpu().shape)

            logits_result = transform_logits(
                upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)

            print("logits shape: ", logits_result.shape)
            res = np.argmax(logits_result, axis=2)
            print("after argmax: ", res.shape, res.max(), res.min())

            for i in range(1, logits_result.shape[2]):
                
                head = np.asarray(res == i).astype(np.uint8) * 255
                #cv2.imshow('head', head)
                #cv2.waitKey(0)

                # Finding Contours
                # Use a copy of the image e.g. edged.copy()
                # since findContours alters the image
                contours, hierarchy = cv2.findContours(head.copy(),
                                                    cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                minimum_countour_lenght = 3
                
                filtered_contour = [c for c in contours if len(c) >= minimum_countour_lenght]
                
                print("class: {} has {} contours:".format(i, len(filtered_contour)))
                for c in filtered_contour:
                    print("\t",len(c))
                    
                output_img = np.zeros((res.shape[0], res.shape[1],3), dtype=np.uint8)
                
                cv2.drawContours(output_img, filtered_contour, -1, (0, 255, 0), 1)
                cv2.imshow('Contours_{}'.format(i), output_img)
                cv2.waitKey(0)

    return


if __name__ == '__main__':
    main()
