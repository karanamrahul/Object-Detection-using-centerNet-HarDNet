from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from pycocotools import mask as maskUtils

from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
from utils.image import blend_truth_mosaic, get_border_coord, mask2box
import math

class CTDetDataset(data.Dataset):
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i
  
  
  def img_transform(self, img, anns, flip_en=True, scale_lv=2, out_shift=None, crop=None):
    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    if self.opt.keep_res:
      input_h = (height | self.opt.pad) + 1
      input_w = (width | self.opt.pad) + 1
      s = np.array([input_w, input_h], dtype=np.float32)
    else:
      s = [img.shape[1], img.shape[0]]
      input_h, input_w = self.opt.input_h, self.opt.input_w
    
    crop = [0, 0, input_w, input_h] if crop is None else crop
    flipped = False
    rot_en = self.opt.rotate > 0
    rot = crpsh_x = crpsh_y =0
    img_s = [img.shape[1], img.shape[0]]
    
    if self.split == 'train':
      if scale_lv == 2:
        s = np.random.choice([ 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896])
      elif scale_lv == 1:
        s = np.random.choice([ 512, 576, 640, 704, 768, 832])
      else:
        s = np.random.choice([ 192, 256, 320, 384, 448, 512])
      
      distortion = 0.6
      sd = np.random.random()*distortion*2 - distortion + 1
      if img.shape[0] > img.shape[1]:
        s = [s, s*(img.shape[0] / img.shape[1])*sd]
      else:
        s = [s*(img.shape[1] / img.shape[0])*sd, s]

      crpsh_x = max( (s[0] - (crop[2]-crop[0])) / 2, (crop[2]-crop[0])*0.2)
      crpsh_y = max( (s[1] - (crop[3]-crop[1])) / 2, (crop[3]-crop[1])*0.2)

      if flip_en and np.random.random() < self.opt.flip:
        flipped = True
        img = img[:, ::-1, :]
      if rot_en:
        rot = np.random.random()*self.opt.rotate*2 - self.opt.rotate

    elif not self.opt.keep_res:
      s = np.array([input_w, input_h], dtype=np.float32) 

    out_center = [input_w/2, input_h/2] if out_shift is None else out_shift
    out_center[0] += (np.random.random()*2-1) * crpsh_x
    out_center[1] += (np.random.random()*2-1) * crpsh_y
    
    trans_input = get_affine_transform(
      c, img_s, rot, s, out_center)
    trans_inv = get_affine_transform(
      c, img_s, rot, s, out_center, inv=1)
      
    inp = cv2.warpAffine(img, trans_input,
                         (input_w, input_h),
                         flags=cv2.INTER_LINEAR)
    inp = (inp.astype(np.float32) / 255.)
    if self.split == 'train' and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)

    output_h = input_h // self.opt.down_ratio
    output_w = input_w // self.opt.down_ratio
    
    num_objs = min(len(anns), self.max_objs) # max_objs = 50  
    ann_list = []  # list of dicts for each object 
    
    border_xy, border_idx = get_border_coord(trans_inv, width, height, crop) # Get the border Coordinates 
      
    ##################################################################################################
    # Changes made : added a if condition to skip the categories which are not in the class id array #
    ##################################################################################################
    
    for k in range(num_objs): # Loop over the number of objects
      ann = anns[k] 
      if(ann['category_id'] not in list(self.cat_ids.keys())): # Check if the object is in the list of categories to be used 
        continue # If not, skip the object
      bbox = self._coco_box_to_bbox(ann['bbox']) # Convert the COCO bounding box to a list of [x1, y1, x2, y2]
      cls_id = int(self.cat_ids[ann['category_id']]) # Get the class id for the object
      if flipped:
          bbox[[0, 2]] = width - bbox[[2, 0]]
      bbox[:2] = affine_transform(bbox[:2], trans_input)
      bbox[2:] = affine_transform(bbox[2:], trans_input)
      segm  = ann['segmentation']

      # Create bbox from the visible part of objects through segmenation mask
      m = self.coco.annToMask(ann)
      bbox2 = mask2box(m, trans_input, border_xy, border_idx, 
                       flipped, width, height, crop)

      if rot_en:
        bbox = bbox2.astype(np.float32)
      ann_list.append([bbox, cls_id, bbox2])
      
      #end of objs loop
    meta = (c, s)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)
    
    return inp, ann_list, output_w, output_h, meta

  
  def get_img_ann(self, index=None, scale_lv=2, out_shift=None, crop=None, flip_en=True):
    index = np.random.randint(len(self.images)) if index is None else index
    img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    img_path = os.path.join(self.img_dir, file_name)
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    
    anns = self.coco.loadAnns(ids=ann_ids)
    img = cv2.imread(img_path)
    
    return self.img_transform(img, anns, flip_en=flip_en, scale_lv=scale_lv, out_shift=out_shift, crop=crop)


  def mosaic_mix(self, index):
    assert not self.opt.keep_res, 'Error: mosaic augmentation requies fixed input size'
    input_h, input_w = self.opt.input_h, self.opt.input_w
    min_offset = 0.5
    max_offset = 0.8
    cut_x = np.random.randint(int(input_w * min_offset), int(input_w * max_offset))
    cut_y = np.random.randint(int(input_h * min_offset), int(input_h * max_offset))

    out_shift = [ [cut_x/2,            cut_y /2        ], [(input_w - cut_x)/2 + cut_x,            cut_y /2],
                  [cut_x/2, (input_h - cut_y)/2 + cut_y], [(input_w - cut_x)/2 + cut_x, (input_h - cut_y)/2 + cut_y] ]
  
    crop = [ [0,0,      cut_x,           cut_y ], [cut_x, 0,     (input_w - cut_x),           cut_y], 
             [0, cut_y, cut_x,(input_h - cut_y)], [cut_x, cut_y, (input_w - cut_x),(input_h - cut_y)] ]
    
    areas  = [cut_x           * cut_y,           (input_w-cut_x) * cut_y,
              cut_x           * (input_h-cut_y), (input_w-cut_x) * (input_h-cut_y)]
    
    for i in range(4):
      scale_lv = 1 if areas[i]/(input_w*input_h) > 0.2 else 0
      if i == 0:
        out_img, ann, output_w, output_h, meta = \
          self.get_img_ann(index, scale_lv=scale_lv, out_shift=out_shift[i], crop=crop[i])
        ann_list = blend_truth_mosaic (out_img, out_img, ann, input_w, input_h, cut_x, cut_y, i)
      else:
        img, ann, _,_,_ = self.get_img_ann(scale_lv=scale_lv, out_shift=out_shift[i], crop=crop[i])
        ann = blend_truth_mosaic (out_img,     img, ann, input_w, input_h, cut_x, cut_y, i)
        ann_list += ann        
    return out_img, ann_list, output_w, output_h, meta

    
  def __getitem__(self, index):
    img_id = self.images[index]
  
    inp, ann_list, output_w, output_h, meta = self.get_img_ann(index, scale_lv=2)
    
    # TBD: Mosaic augmentation requires large input image size
    # Increase input image size from 512x512 to 800x800 or larger and
    # adjust the scale level to avoid the mosaic boundary to become 
    # a significant boundary of objects
    #inp, ann_list, output_w, output_h, meta = self.mosaic_mix( index )
    
    if False: # Augmnetation visualization
      img = inp.transpose(1, 2, 0)
      img = (img*self.std + self.mean)*255
      for an in ann_list:
        bbox, cls_id, bbox2 = an
        bbox = bbox.astype(np.int32)
        bbox2 = bbox2.astype(np.int32)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, img.shape[1])
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, img.shape[0])
        bbox2[[0, 2]] = np.clip(bbox2[[0, 2]], 0, img.shape[1])
        bbox2[[1, 3]] = np.clip(bbox2[[1, 3]], 0, img.shape[0])
        if bbox[2] - bbox[0] > 0 and bbox[3] - bbox[1] > 0:
          cv2.rectangle(img, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (255,0,0), 3)
        if bbox2.shape[0] > 0:
          cv2.rectangle(img, (bbox2[0],bbox2[1]), (bbox2[2],bbox2[3]), (0,255,0), 2)
      cv2.imwrite('temp_%d.jpg'%(index),img)
    
    num_objs = min(len(ann_list), self.max_objs)
    num_classes = self.num_classes
    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    dense_reg = np.zeros((4, output_h, output_w), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    dense_wh_mask = np.zeros((4, output_h, output_w), dtype=np.float32)
    cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
    cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)
    
    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian

    gt_det = []
    xs = np.random.randint(output_w, size=(self.max_objs, 1))
    ys = np.random.randint(output_h, size=(self.max_objs, 1))
    bgs = np.concatenate([xs,ys], axis=1)
    
    for k in range(num_objs):
      bbox, cls_id, bbox2 = ann_list[k]
      
      bbox /= self.opt.down_ratio
      bbox2 /= self.opt.down_ratio

      oh, ow = bbox[3] - bbox[1], bbox[2] - bbox[0]
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      
      if (h/(oh+0.01) < 0.9 or  w/(ow+0.01) < 0.9) and bbox2.shape[0] > 0:
        bbox = bbox2
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h)
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      #get center of box
      ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
      ct_int = ct.astype(np.int32)

      if (h > 2 or h/(oh+0.01) > 0.5) and (w > 2 or w/(ow+0.01) > 0.5):
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        radius = self.opt.hm_gauss if self.opt.mse_loss else radius

        draw_dense_reg(dense_reg, dense_wh_mask, ct_int, bbox, radius)
        draw_gaussian(hm[cls_id], ct_int, radius)

        cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
        cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
        gt_det.append([ct[0] - w / 2, ct[1] - h / 2, 
                       ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
    
    dense_wh  = dense_reg[:2,:,:]
    dense_off = dense_reg[2:,:,:]

    ret = {'input': inp, 'hm': hm, 'dense_wh': dense_wh, 'dense_off': dense_off, 
           'dense_wh_mask': dense_wh_mask[:2]}
    if self.opt.cat_spec_wh:
      ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
      del ret['wh']
    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 6), dtype=np.float32)
      meta = {'c': meta[0], 's': meta[1], 'gt_det': gt_det, 'img_id': img_id}
      ret['meta'] = meta
    return ret
    
