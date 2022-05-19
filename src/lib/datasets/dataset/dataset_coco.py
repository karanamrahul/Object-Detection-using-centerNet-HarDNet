
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data


###############################################################################
#           Creating Custom Dataset using COCO                            #
###############################################################################

"""
 *  MIT License
 *
 *  Copyright (c) 2022 Rahul Karanam
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a
 *  copy of this software and associated documentation files (the "Software"),
 *  to deal in the Software without restriction, including without
 *  limitation the rights to use, copy, modify, merge, publish, distribute,
 *  sublicense, and/or sell copies of the Software, and to permit persons to
 *  whom the Software is furnished to do so, subject to the following
 *  conditions:
 *
 *  The above copyright notice and this permission notice shall be included
 *  in all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 *  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *  DEALINGS IN THE SOFTWARE.
"""

"""
 @file       dataset_coco.py
 @author     Rahul Karanam
 @copyright  MIT License
 @brief      This file contains the main function for creating custom dataset using COCO.
 
"""

class cocoDataset(data.Dataset):
  num_classes = 5 # As we need only [person,car,truck,dog,cat] classes for our dataset, we are not using the coco.loadImgs() function 
  default_resolution = [512, 512] #
  mean = np.array([0.40789654, 0.44719302, 0.47026115], 
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

  def __init__(self, opt, split):
    super(cocoDataset, self).__init__()
    self.data_dir = os.path.join(opt.data_dir, 'coco')
    self.img_dir = os.path.join(self.data_dir, '{}2017'.format(split))
    if split == 'test':
      self.annot_path = os.path.join(
          self.data_dir, 'annotations', 
          'image_info_test-dev2017.json').format(split)
    else:
      if opt.task == 'exdet':
        self.annot_path = os.path.join(
          self.data_dir, 'annotations', 
          'instances_extreme_{}2017.json').format(split)
      else:
        self.annot_path = os.path.join(
          self.data_dir, 'annotations', 
          'instances_{}2017.json').format(split)

###############################################################################
# We make the following changes to the original COCO dataset:              #
###############################################################################
    self.max_objs = 128
    self.class_name = [
      '__background__','car','person','truck', 'cat', 'dog'] # We add car,person,truck, cat, & dog classes for our dataset along with the id of each class
    self._valid_ids = [3,1,8,17,18]


    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)} # We make a dictionary of the class id and the class name 
    

    self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                      for v in range(1, self.num_classes + 1)] # We make a list of the colors for each class 
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)


    self.split = split # We make a variable to store the split of the dataset
    self.opt = opt  # We make a variable to store the options of the dataset

    print('==> initializing coco 2017 {} data.'.format(split)) # We print the split of the dataset
    self.coco = coco.COCO(self.annot_path) # We make a variable to store the COCO dataset 
 
    # here youll pull the ids of the images that you need
    # self.images = self.coco.getImgIds()


    img_ids = []
    for id in self._valid_ids:
        img_ids.extend(self.coco.getImgIds(catIds=[id]))  # We make a variable to store the image ids of the dataset 
    

    self.images = list(set(img_ids)) # We make a variable to store the image ids of the dataset


    self.num_samples = len(self.images) # We make a variable to store the number of samples in the dataset

    print('Loaded {} {} samples .... '.format(split, self.num_samples)) # We print the number of samples in the dataset

  def _to_float(self, x): # We make a function to convert the data to float
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes): # We make a function to convert the data to the evaluation format 

    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = self._valid_ids[cls_ind - 1]
        for bbox in all_bboxes[image_id][cls_ind]:
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = bbox[4]
          bbox_out  = list(map(self._to_float, bbox[0:4]))

          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(score))
          }
          if len(bbox) > 5:
              extreme_points = list(map(self._to_float, bbox[5:13]))
              detection["extreme_points"] = extreme_points
          detections.append(detection)
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results), 
                open('{}/results.json'.format(save_dir), 'w'))
  
  def run_eval(self, results, save_dir):
    self.save_results(results, save_dir)
    coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
