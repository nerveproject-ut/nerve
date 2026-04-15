
import os
import cv2
import numpy as np

from pycocotools.coco import COCO
from yoloX.yolox.data.datasets.coco import COCODataset, remove_useless_info

class DVS_Radar_Dataset(COCODataset):
    def __init__(self, data_dir:str, json_file:str, name:str, img_size:tuple, preproc=None, cache=False, cache_type="ram", 
                 enable_mosaic=False, use_also_radar=False, include_distance=False, min_dist=0.0, max_dist=10.0):
        self.include_distance = include_distance
        self.use_also_radar = use_also_radar
        self.min_distance = min_dist
        self.max_distance = max_dist

        
        assert not data_dir is None
        self.data_dir = data_dir
        self.json_file = json_file

        self.coco = COCO(os.path.join(self.data_dir, "coco_labels", self.json_file))
        remove_useless_info(self.coco)
        self.class_ids = sorted(self.coco.getCatIds())

        self.img_size = img_size

        # as first, let's load all annotations.
        self.ids = self.coco.getImgIds()
        self.annotations = self._load_coco_annotations()
        
        #now, if we are using distance, let's filter out data where distance is outside our interest range.
        if self.include_distance:
            idx_to_be_removed = []
            for idx, item in enumerate(self.annotations):
                (ann, _, _, _) = item
                for subject in range(len(ann)):
                    dist = ann[subject, 5]
                    if dist < self.min_distance or dist > self.max_distance:
                        # in this frame there is at least one subject with unknown distance, or which is too close.
                        # let's discard this frame.
                        idx_to_be_removed.append(idx)
                        break
            if len(idx_to_be_removed) > 0:
                print("{} frames (with annotations) will be discarded, having distance information out of range.".format(len(idx_to_be_removed)))
                idx_to_be_removed = sorted(idx_to_be_removed, reverse=True)
                for idx in idx_to_be_removed:
                    self.ids.pop(idx)
                    self.annotations.pop(idx)
                
            assert len(self.ids) == len(self.annotations)


        self.num_imgs = len(self.ids)
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in self.cats])
        self.name = name
        self.preproc = preproc
        

        path_filename = [os.path.join(name, anno[3]) for anno in self.annotations]
        #let's invoke the constructor of the parent of COCODataset, bypassing the constructor of COCODataset
        super(COCODataset, self).__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=data_dir,
            cache_dir_name=f"cache_{name}",
            path_filename=path_filename,
            cache=cache,
            cache_type=cache_type
        )
        self.enable_mosaic = enable_mosaic
        
    def load_image(self, index):
        file_name = self.annotations[index][3]
        img_file = os.path.join(self.data_dir, self.name, file_name)

        img = cv2.imread(img_file)
        assert img is not None, f"file named {img_file} not found"

        if not self.use_also_radar:
            return img
        

        # in this case, let's fetch the corresp. radar frame, and merge them in the third channel.
        # The 'third channel' in this case is the Rs channel, since data were stored in rgb but here are loaded as bgr.
        # Nevertheless, it should not change which channel to 'drop', since one out of three is a linear combination of the other two.

        # example of radar file format: point_cloud_img__0000000171.png
        file_number = str(file_name).split('.')[0]
        file_number = file_number.split('_')[-1]
        radar_poc_path = os.path.join(self.data_dir, self.name, 'ti_radar', 'point_cloud_img__{}.png'.format(file_number))
        assert os.path.isfile(radar_poc_path), "File {} was not found.".format(radar_poc_path)
        radar_frame = cv2.imread(radar_poc_path) # in the filesystem, this frame was stored w 3 channels, but actually they are all equal. We are interested in just a channel.

        radar_frame = cv2.dilate(radar_frame, np.ones((5, 5), np.uint8)) #let's perform a little bit of dilatation, so that during shrinking we don't loose these pixels.


        img[:,:,-1] = radar_frame[:,:,0] # FUSION
        #print(f'Loaded img {img_file}')
        return img
    
    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint8)
        return resized_img
    
    def load_anno_from_ids(self, id_):
        if not self.include_distance:
            return super().load_anno_from_ids(id_)

        # let's add the subject distance as label

        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 6))        # changed from 5 to 6
        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls
            res[ix, 5] = obj["avg_distance"] # added this line --> distance GT

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return (res, img_info, resized_info, file_name)

