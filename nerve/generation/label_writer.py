import os
import sys
import torch
import json
import copy

from nerve.extraction.utils.cameraParams import Mapping
from nerve.extraction.utils.dataset_utils import GetCategories, GetFilteredCategories
from nerve.extraction.mapping.mapping_utils import MapLabels



class LabelWriter():
    def __init__(self, path:str, filter_class_ids: list = None) -> None:
        """
        Initialize the LabelWriter.
        
        Args:
            path: Path to the output JSON annotation file
            filter_class_ids: Optional list of COCO category IDs to include.
                             If provided, only these categories will be in the output JSON.
                             Original COCO IDs are preserved (not remapped) for compatibility
                             with pre-trained models and COCO evaluation tools.
                             If None, all COCO categories are included.
        """
        assert path.endswith('.json')
        self.out_path = path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set up category filtering (keeping original COCO IDs)
        self.filter_class_ids = set(filter_class_ids) if filter_class_ids else None
        if filter_class_ids is not None and len(filter_class_ids) > 0:
            self.categories = GetFilteredCategories(filter_class_ids)
        else:
            self.categories = GetCategories()

        if os.path.isfile(path):
            #if the file already exists, then let's access it and continue it.
            print("Found existing annotations at {}, new data will be appeneded.".format(path))
            with open(path, 'r') as openfile:
                self.json_file = json.load(openfile)
                # Handle case where annotations list is empty (no objects found in previous sessions)
                if len(self.json_file['annotations']) > 0:
                    self.annotation_counter = self.json_file['annotations'][-1]['id'] + 1
                else:
                    self.annotation_counter = 1
        else:
            self.json_file = {
                "info": {
                    "description": "NERVE multi-sensor office dataset",
                    "url": "https://data.4tu.nl/",
                    "version": "1.0",
                    "year": 2024,
                    "contributor": "NERVE",
                    "date_created": "2024/01/01"},
                "licenses": [
                    {
                        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                        "id": 1,
                        "name": "Attribution-NonCommercial-ShareAlike License"
                    }],
                "images": [],
                "annotations": [],
                "categories": self.categories
            }
            self.annotation_counter = 1

    def get_last_image_index(self):
        if len(self.json_file['images']) > 0:
            return self.json_file['images'][-1]['id']
        return -1  # Return -1 so that current_index starts at 0

    def write_file(self):
        if os.path.isfile(self.out_path):
            os.remove(self.out_path)
        with open(self.out_path, "w") as write_file:
            json.dump(self.json_file, write_file)

    def add_data(self, annotations:list, mapping:Mapping, data_path:str, frame_size:tuple, image_id:int, transformation_function=None):
        
        if not annotations is None and len(annotations) > 0:
            anns = MapLabels(annotations, mapping, None, self.device)
            for a in anns:
                # Skip annotations for categories not in our filter (if filtering is enabled)
                # Note: category_id is kept as original COCO ID - no remapping
                if self.filter_class_ids is not None:
                    if a['category_id'] not in self.filter_class_ids:
                        continue
                
                a['image_id'] = image_id # correcting the image_id with the new one
                a['id'] = self.annotation_counter
                
                if not transformation_function is None:
                    a = transformation_function(a)
                self.json_file["annotations"].append(copy.deepcopy(a))
                self.annotation_counter += 1

        height, width = int(frame_size[0]), int(frame_size[1])
        self.json_file['images'].append({
            'id':image_id,
            'license':1,
            'file_name':str(data_path),
            'width':width,
            'height':height
        })
