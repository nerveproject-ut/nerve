from pathlib import Path
from pycocotools.coco import COCO
import numpy as np
import cv2
import json


class CustomCOCO(COCO):
    def __init__(self, annotation_file=None) -> None:
        super().__init__(annotation_file)
        if annotation_file is None:
            self.dataset = {
                'info':{},
                'licenses':[],
                'images':[],
                'annotations':[],
                'categories':[]
            }
        else:
            self.fps = self.dataset['info']['fps']
            self.frame_width = self.dataset['info']['frame_width']
            self.frame_height = self.dataset['info']['frame_height']
            self.totalImages = len(self.imgs)
            
            self.entity_ids={}
            self.number_entities = 0
            for ann in self.anns.values():
                id = ann['track_id']
                if id in self.entity_ids:
                    self.entity_ids[id] += 1
                else:
                    self.number_entities += 1
                    self.entity_ids[id] = 0
                    
            print('Dataset info:',
                '\n\t frame HxW @ fps : {}x{} @ {}'.format(self.frame_height, self.frame_width, self.fps),
                '\n\t number of annotations : {}'.format(len(self.anns)),
                '\n\t number of different entities : {}'.format(self.number_entities),
                '\n\t number of frames : {}'.format(self.totalImages),
                '\n\t total video time: {:.1f} seconds'.format(self.totalImages / self.fps),
                )
    
    def retrieve_imageIDs_from_time(self, predictions)->list:
        if type(predictions) == str:
            with open(predictions) as f:
                anns = json.load(f)
        else:
            assert type(predictions) == list
            anns = predictions

        all_imgs = list(self.imgs.values())

        img0 = all_imgs[0]
        t0 = img0['time_ms']
        period_ms = 1e3/self.fps
        # For how the dataset is structured, we know that all next labels will have be located at t = t0 + idx * period_ms

        for ann in anns:
            time_ms = ann['time_ms']
            idx = round((time_ms - t0)/period_ms)
            target = all_imgs[idx]
            assert abs(target['time_ms'] - time_ms) <= period_ms/2

            ann['image_id'] = target['id']

        return anns



        

        

class VisualCOCO(CustomCOCO):
    def __init__(self, annotation_file=None) -> None:
        super().__init__(annotation_file)

        self.color_dark_factor = 75 # just for visualization
        self.color_variations_cache = {}
        self.entity_colors={}

        for id in self.entity_ids:
            self.entity_colors[id] = np.random.random_integers(size=(3,), low=0, high=155)+100

    def getColorVariations(self, n_variations:int):
        if n_variations in self.color_variations_cache:
            return self.color_variations_cache[n_variations]
        
        mat = np.random.rand(n_variations, 4)
        mat[-1] = 0
        mat = np.power(mat,1.5)
        mat = mat / (np.linalg.norm(mat,axis=-1, keepdims=True) + 1e-16)
        mat = (mat * self.color_dark_factor).astype(np.uint8)
        self.color_variations_cache[n_variations] = mat
        return mat
    

    def getAnnotatedFrame(self, imgID: int, anns: list = None, 
                          show_bb: bool = True, show_conf:bool=False, show_distance:bool=False,
                          show_ID: bool = False, show_class:bool = False, 
                          show_entity_seg: bool = False, show_skeleton: bool = False, 
                          show_body_parts_seg: bool = False, skeleton_conf_thres:float=0.5, 
                          show_image_id:bool = False, show_annotation_id:bool=False, show_distance_points=False):
        """
        Get a numpy array displaying the given annotations with the given settings.
        If no annotation is passed (anns==None), then all the annotations linked to the given imgID will be loaded.
        if specific annotations are passed, they MUST match the given imgID
        :param anns (array of object): annotations to display
        :return: np.ndarray
        """
        
        assert imgID in self.imgs, "There are no annotations about imgID {}".format(imgID)
        
        if anns is not None:
            for a in anns:
                assert a['image_id'] == imgID, 'all the given annotations must match the given imgID ({})'.format(imgID)
        else:
            annIds = self.getAnnIds(imgIds=[imgID])
            anns = self.loadAnns(annIds)
            #print('loaded {} annotations.'.format(len(anns)))
            
        # RGBA
        img = np.zeros((self.frame_height, self.frame_width, 4), dtype=np.int32)
        
        black_color=(0,0,0,255)

        for ann in anns:
            ann_color = self.entity_colors[ann['track_id']]
            darker_color = (ann_color - self.color_dark_factor).tolist()
            darker_color.append(128)
            base_color = ann_color.tolist()
            base_color.append(255)
            
            if show_entity_seg and 'segmentation' in ann:
                for seg in ann['segmentation']:
                    poly = np.array(seg,dtype=np.int32).reshape((int(len(seg)/2), 2))
                    cv2.fillPoly(img, pts=[poly], color=darker_color)
                    cv2.polylines(img, pts=[poly], isClosed=True, color=base_color, thickness=2)
                    
            if show_body_parts_seg and 'parts' in ann:
                parts = ann['parts']
                sectors = len(parts)
                color_variations = self.getColorVariations(sectors)
                for idx, sector in enumerate(parts):
                    sec_color = (darker_color + color_variations[idx]).tolist()
                    for seg in sector:
                        poly = np.array(seg,dtype=np.int32).reshape((int(len(seg)/2), 2))
                        cv2.fillPoly(img, pts=[poly], color=sec_color)
                        cv2.polylines(img, pts=[poly], isClosed=True, color=base_color, thickness=2)
                    
            if (show_bb or show_ID or show_class or show_conf or show_annotation_id) and 'bbox' in ann:
                [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
                poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
                poly = np.array(poly, dtype=np.int32).reshape((4,2))
                cv2.polylines(img, pts=[poly], isClosed=True, color=base_color, thickness=3)
                
                if show_conf and 'conf' in ann:
                    text_conf = '{:.2f}'.format(ann['conf'])
                    n_chars = len(text_conf)
                    w,h = 9*n_chars + 10, 25
                    x,y = int(bbox_x),int(bbox_y +bbox_h)-h
                    cv2.rectangle(img, (x,y), (x + w, y + h), color=base_color, thickness=-1)
                    cv2.putText(img=img, text=text_conf,org=(x+5 ,y+20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=black_color, thickness=1)
                
                if show_ID and 'track_id' in ann:
                    text_id = str(ann['track_id'])
                    n_chars = len(text_id)
                    w,h = 20*n_chars + 10, 25
                    x,y = int(bbox_x),int(bbox_y)
                    cv2.rectangle(img, (x,y), (x + w, y + h), color=base_color, thickness=-1)
                    cv2.putText(img=img, text=text_id,org=(x+5 ,y+20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=black_color, thickness=2)
                
                if show_class:
                    class_id = ann['category_id']
                    cl = self.cats[class_id]['name']
                    n_chars = len(cl)
                    w, h = 9*n_chars + 10, 25
                    x,y= int(bbox_x+bbox_w)-w,int(bbox_y)
                    cv2.rectangle(img, (x,y), (x + w, y + h), color=base_color, thickness=-1)
                    cv2.putText(img=img, text=cl,org=(x+5 ,y+15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=black_color, thickness=1)

                if show_annotation_id :
                    text_ann_id = '{}'.format(ann['id'])
                    n_chars = len(text_ann_id)
                    w,h = 9*n_chars + 10, 25
                    x,y = int(bbox_x+bbox_w)-w,int(bbox_y +bbox_h)-h
                    cv2.rectangle(img, (x,y), (x + w, y + h), color=base_color, thickness=-1)
                    cv2.putText(img=img, text=text_ann_id,org=(x+5 ,y+20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=black_color, thickness=1)
            
            if show_distance and 'avg_distance' in ann:
                text_dist = '{:.2f}'.format(ann['avg_distance'])
                n_chars = len(text_dist)
                w,h = 9*n_chars + 10, 25
                x,y = int(bbox_x),int(bbox_y +bbox_h)-h
                cv2.rectangle(img, (x,y), (x + w, y + h), color=base_color, thickness=-1)
                cv2.putText(img=img, text=text_dist, org=(x+5 ,y+20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=black_color, thickness=1)

            if show_skeleton and 'keypoints' in ann:
                # turn skeleton into zero-based index
                sks = np.array(self.loadCats(ann['category_id'])[0]['skeleton'])-1
                kp = np.array(ann['keypoints'])
                if len(kp) > 0:
                    x = kp[0::3].astype(np.int32)
                    y = kp[1::3].astype(np.int32)
                    c = kp[2::3]
                    for sk in sks:
                        if np.all(c[sk]>skeleton_conf_thres):
                            a = (x[sk[0]], y[sk[0]])
                            b = (x[sk[1]], y[sk[1]])
                            cv2.line(img, a, b, color=base_color, thickness=2)
                            cv2.circle(img, a, radius=5, color=base_color, thickness=-1)
                            cv2.circle(img, b, radius=5, color=base_color, thickness=-1)
            
            if show_distance_points and 'distance_points' in ann:
                samples = ann['distance_points']
                for i in range(len(samples)//3):
                    x, y = int(samples[3*i]), int(samples[3*i+1])
                    cv2.circle(img, (x,y), radius=5, color=base_color, thickness=2)

            
        
        if show_image_id:
            text = "img_id: {}".format(imgID)
            n_chars = len(text)
            w,h = 12*n_chars + 15, 25
            x,y = 0, img.shape[0] -h
            cv2.rectangle(img, (x,y), (x + w, y + h), color=(0, 255, 0, 255), thickness=-1)
            cv2.putText(img=img, text=text,org=(x+5 ,y+20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=black_color, thickness=1)
            
        return img.astype(np.uint8)
