import numpy as np
import cv2
import torch
import cv2

import torchvision.transforms as transforms

import nerve.extraction.segmentation.networks as networks
from .utils.transforms import transform_logits, get_affine_transform

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


class BodySegmenter:
    def __init__(self, weights_path: str, dataset: str = 'pascal',
                 avoid_background: bool = True, min_contour_size: int = 6) -> None:

        self.avoid_background = avoid_background
        self.min_contour_size = min_contour_size

        self.input_size = dataset_settings[dataset]['input_size']
        self.num_classes = dataset_settings[dataset]['num_classes']
        self.classes = dataset_settings[dataset]['label']

        self.aspect_ratio = self.input_size[1] * 1.0 / self.input_size[0]

        self.model = networks.init_model(
            'resnet101', num_classes=self.num_classes, pretrained=None)

        state_dict = torch.load(weights_path)['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        self.model.cuda()
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[
                0.225, 0.224, 0.229])
        ])

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w, h], dtype=np.float32)
        return center, scale

    def get_class_names(self):
        if not self.avoid_background:
            return self.classes
        else:
            return self.classes[1:]

    def __call__(self, img: np.ndarray, mask_filter: np.ndarray = None, add_x_offset:int = 0, add_y_offset:int = 0):
        with torch.no_grad():
            h, w, _ = img.shape
            person_center, s = self._box2cs([0, 0, w - 1, h - 1])
            person_center = np.array(person_center)
            s = np.array(s)
            r = 0
            trans = get_affine_transform(person_center, s, r, self.input_size)
            input = cv2.warpAffine(
                img,
                trans,
                (int(self.input_size[1]), int(self.input_size[0])),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0))

            input = self.transform(input)
            input = input.unsqueeze(0)

            output = self.model(input.cuda())
            upsample = torch.nn.Upsample(
                size=self.input_size, mode='bilinear', align_corners=True)
            upsample_output = upsample(output[0][-1][0].unsqueeze(0))
            upsample_output = upsample_output.squeeze()
            upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

            logits_result = transform_logits(
                upsample_output.data.cpu().numpy(), person_center, s, w, h, input_size=self.input_size)

            res = np.argmax(logits_result, axis=2)

            if mask_filter is not None:
                res = res * mask_filter

            min_idx = 1 if self.avoid_background else 0

            vertices = []
            for i in range(min_idx, logits_result.shape[2]):
                mask = np.asarray(res == i).astype(np.uint8) * 255

                # Finding Contours
                # Use a copy of the image e.g. edged.copy()
                # since findContours alters the image
                contours, hierarchy = cv2.findContours(mask,
                                                       cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                class_vertices = []
                for i in range(len(contours)):
                    if len(contours[i]) <= self.min_contour_size:
                        continue
                    shape_vertices = []
                    for c in contours[i]:
                        shape_vertices.append(float(c[0][0]) + add_x_offset)
                        shape_vertices.append(float(c[0][1]) + add_y_offset)
                    class_vertices.append(shape_vertices)

                vertices.append(class_vertices)
                # output_img = np.zeros((res.shape[0], res.shape[1], 3), dtype=np.uint8)
                # cv2.drawContours(output_img, filtered_contour, -1, (0, 0, 255), 1)
                # cv2.imshow('Contours_{}'.format(i), output_img)
                # cv2.waitKey(0)
            return vertices
