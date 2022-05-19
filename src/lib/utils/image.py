# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import random

def flip(img):
  return img[:, :, ::-1].copy()  

def transform_preds(coords, center, scale, output_size):
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    target_coords = affine_transform(coords.T, trans).T
    return target_coords


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=None,
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)
    if shift is None:
      shift=np.array([output_size[0]/2, output_size[1]/2], dtype=np.float32)
    else:
      shift=np.array([shift[0], shift[1]], dtype=np.float32)
    scale_tmp = scale
    src_w = scale_tmp[0]
    src_h = scale_tmp[1]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_h * -0.5], rot_rad)
    dst_dir = np.array([0, dst_h * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center 
    src[1, :] = center + src_dir 
    src[2, :] = center + get_dir([src_w * -0.5, 0], rot_rad)

    dst[0, :] = shift 
    dst[1, :] = shift + dst_dir
    dst[2, :] = shift + np.array([dst_w * -0.5, 0], np.float32)
    
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    if pt.ndim == 2 and pt.shape[0] == 2:
      new_pt = np.concatenate([pt, np.ones((1,pt.shape[1]))], axis=0)
      new_pt = np.dot(t, new_pt)
      return new_pt[:2,:]
    else:
      new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
      new_pt = np.dot(t, new_pt)
      return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img,
                             trans,
                             (int(output_size[0]), int(output_size[1])),
                             flags=cv2.INTER_LINEAR)

    return dst_img


def gaussian_radius(det_size, min_overlap=0.7):
  height, width = det_size

  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2
  return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
  
  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]
    
  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  if x + right <= 0 or x - left >= width or y + bottom <= 0 or y - top >= height:
   return heatmap

  masked_heatmap  = heatmap[ y - top:y + bottom, x - left:x + right ]
  masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
  
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
  return heatmap


def draw_dense_reg(regmap, regmask, center, bbox, radius):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 24)
  gaussian_wt = gaussian / gaussian.sum()
  gaussian_wt = np.repeat(gaussian_wt.reshape(1,diameter, diameter), 4, axis=0)

  ct  = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
  wh  = np.array([ bbox[2] - bbox[0], bbox[3] - bbox[1]], dtype=np.float32).reshape(2, 1, 1)
  xs = np.repeat(np.arange( diameter ).reshape(1,1,-1), diameter, axis=1)-radius+center[0]
  ys = np.repeat(np.arange( diameter ).reshape(1,-1,1), diameter, axis=2)-radius+center[1]

  reg = np.ones((2, diameter, diameter), dtype=np.float32) * wh
  off = np.ones((2, diameter, diameter), dtype=np.float32) * ct.reshape(2, 1, 1) - np.concatenate([xs,ys], axis=0)
  reg = np.concatenate([reg,off], axis=0)
  
  x, y = int(center[0]), int(center[1])

  height, width = regmap.shape[1:3]
    
  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)
  
  if x + right <= 0 or x - left >= width or y + bottom <= 0 or y - top >= height:
    return regmap, regmask
    
  masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
  masked_reg = reg[:, radius - top:radius + bottom,
                      radius - left:radius + right]

  masked_regmask = regmask[:,y - top:y + bottom, x - left:x + right]
  masked_gaussian_wt = gaussian_wt[:, radius - top:radius + bottom,
                                      radius - left:radius + right]

  if min(masked_gaussian_wt.shape) > 0 and min(masked_regmask.shape) > 0: 
    
    idx = (masked_gaussian_wt >= masked_regmask)
    masked_regmap  = (1-idx) * masked_regmap  + idx * masked_reg
    masked_regmask = (1-idx) * masked_regmask + idx * masked_gaussian_wt

  regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
  regmask[:,  y - top:y + bottom, x - left:x + right] = masked_regmask
  return regmap, regmask


def draw_msra_gaussian(heatmap, center, sigma):
  tmp_size = sigma * 3
  mu_x = int(center[0] + 0.5)
  mu_y = int(center[1] + 0.5)
  w, h = heatmap.shape[0], heatmap.shape[1]
  ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
  br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
  if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
    return heatmap
  size = 2 * tmp_size + 1
  x = np.arange(0, size, 1, np.float32)
  y = x[:, np.newaxis]
  x0 = y0 = size // 2
  g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
  g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
  g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
  img_x = max(0, ul[0]), min(br[0], h)
  img_y = max(0, ul[1]), min(br[1], w)
  heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
    g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
  return heatmap

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3, ))
    image += np.dot(eigvec, eigval * alpha)

def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2

def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])

def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha

def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)

def color_aug(data_rng, image, eig_val, eig_vec):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)
    lighting_(data_rng, image, 0.1, eig_val, eig_vec)


def filter_truth(anns, sx, sy, xd, yd):
    ann_list = []
    for i in range(len(anns)):
      bbox, cls_id = anns[i]
      bbox[ 0] = np.clip(bbox[ 0], sx, xd)
      bbox[ 2] = np.clip(bbox[ 2], sx, xd)

      bbox[ 1] = np.clip(bbox[ 1], sy, yd)
      bbox[ 3] = np.clip(bbox[ 3], sy, yd)
      if (bbox[ 2] - bbox[ 0]) > 1 and (bbox[ 3] - bbox[ 1]) > 1:
        ann_list.append( [bbox, cls_id] )

    return ann_list


def blend_truth_mosaic(out_img, img, anns, w, h, cut_x, cut_y, i_mixup):

    if i_mixup == 0:
        anns = filter_truth(anns, 0, 0, cut_x, cut_y)
        out_img[:,:cut_y, :cut_x] = img[:,:cut_y, :cut_x]
    if i_mixup == 1:
        anns = filter_truth(anns, cut_x, 0, w, cut_y)
        out_img[:,:cut_y, cut_x:] = img[:,:cut_y, cut_x:]
    if i_mixup == 2:
        anns = filter_truth(anns, 0, cut_y, cut_x, h)
        out_img[:,cut_y:, :cut_x] = img[:,cut_y:, :cut_x]
    if i_mixup == 3:
        anns = filter_truth(anns, cut_x, cut_y, w, h)
        out_img[:,cut_y:, cut_x:] = img[:,cut_y:, cut_x:]

    return anns


def get_border_coord(trans_inv, width, height, crop):
    '''
    Inverse project the border line of training window (512x512) back to
    the original input image space, then return the coordinates of 
    the iversed-projected border line
    '''
    x0, y0, xe, ye = crop
    bxs = np.concatenate([np.arange(x0,xe), np.arange(x0,xe), np.repeat(x0, ye-y0), np.repeat(xe-1, ye-y0)])
    bys = np.concatenate([np.repeat(y0, xe-x0), np.repeat(ye-1, xe-x0), np.arange(y0,ye), np.arange(y0,ye)])
    border = affine_transform(np.array([bxs, bys]), trans_inv)

    x_in = np.logical_and(border[0,:]>=0, border[0,:]<width)
    y_in = np.logical_and(border[1,:]>=0, border[1,:]<height)
    border = border[:, np.logical_and(x_in, y_in)]
    border_idx  = border[0,:].astype(np.int32) + \
                  border[1,:].astype(np.int32) * (width+1)

    return border, border_idx.astype(np.int32)
    

def mask2box(mask, trans, border, border_idx, flipped, width, height, crop):
    '''
    Extract the edge of an object from the segmentation mask, then project it
    to the training window (512x512), apply cropping such that only the visible
    part of the edge are going to be preserved. Then, combine with the overlap
    between object and training window border to create a bbox that perfectly
    fits the visible part of the target object
    '''
    x0, y0, xe, ye = crop

    if flipped:
      mask = np.flip(mask, 1)

    # add a column of zero to avoid zero edge for full-image large objects
    z = np.repeat([[0]], mask.shape[0], axis=0)
    mask = np.concatenate([ mask, z], 1)
    mask = mask.flatten()
    # Extract edges of the object
    pixels = np.concatenate([[0], mask, [0]])
    pts = np.where(pixels[1:] != pixels[:-1])[0]
    xs = pts %  (width+1)
    ys = pts // (width+1)

    # Extract the overlap between the object segment and the border line of
    # the training window that was inverse projected back to the original image space
    border_mask = mask[ border_idx ]
    border_pts  = border[ :, np.where(border_mask == 1)[0] ]

    # Project to training window
    segm_pts   = affine_transform(np.array([xs, ys]),  trans)
    border_pts = affine_transform(border_pts,          trans)
    
    x_in = np.logical_and(segm_pts[0,:]>=x0-1, segm_pts[0,:]<xe+1)
    y_in = np.logical_and(segm_pts[1,:]>=y0-1, segm_pts[1,:]<ye+1)
    segm_pts = segm_pts[:, np.logical_and(x_in, y_in)]
    segm_pts = np.concatenate([segm_pts, border_pts],axis=1)

    if segm_pts.shape[1] > 0:
      return  np.array([segm_pts[0,:].min(), segm_pts[1,:].min(),
                        segm_pts[0,:].max(), segm_pts[1,:].max()], dtype=np.float32)
    else:
      return  np.array([0,0,0,0], dtype=np.float32)

    