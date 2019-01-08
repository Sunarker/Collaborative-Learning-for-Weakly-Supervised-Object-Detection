# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
try:
  import cPickle as pickle
except ImportError:
  import pickle
import os
import math

from utils.timer import Timer
from model.nms_wrapper import nms
from utils.blob import im_list_to_blob

from model.config import cfg, get_output_dir
from model.bbox_transform import clip_boxes, bbox_transform_inv

import torch

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.
    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob
    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois_blob_real = []

    for i in range(len(im_scale_factors)):
        rois, levels = _project_im_rois(im_rois, np.array([im_scale_factors[i]]))
        rois_blob = np.hstack((levels, rois))
        rois_blob_real.append(rois_blob.astype(np.float32, copy=False))

    return rois_blob_real


def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.
    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob
    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1
        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels


def _get_blobs(im, rois):
  """Convert an image and RoIs within that image into network inputs."""
  blobs = {}
  blobs['data'], im_scale_factors = _get_image_blob(im)
  blobs['boxes'] = _get_rois_blob(rois, im_scale_factors)
  
  return blobs, im_scale_factors

def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes

def _rescale_boxes(boxes, inds, scales):
  """Rescale boxes according to image rescaling."""
  for i in range(boxes.shape[0]):
    boxes[i,:] = boxes[i,:] / scales[int(inds[i])]

  return boxes

def im_detect(net, im, boxes):
  blobs, im_scales = _get_blobs(im,boxes)
  assert len(im_scales) == 1, "Only single-image batch implemented"

  im_blob = blobs['data']
  blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)
  cfg.DEDUP_BOXES=1.0/16.0
  for i in range(len(blobs['data'])):
    if cfg.DEDUP_BOXES > 0:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['boxes'][i] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['boxes'][i] = blobs['boxes'][i][index, :]
        boxes_tmp = boxes[index, :].copy()
    else:
        boxes_tmp = boxes.copy()
    
    # TODO
    # change the blobs['im_info'], now is an array
    cls_prob, bbox_prob, fuse_prob, image_prob, scores_fast, bbox_pred_fast, rois = net.test_image(blobs['data'][i:i+1,:], blobs['im_info'], blobs['boxes'][i])
    '''
       WSDDN
    '''
    scores_tmp = fuse_prob
    pred_boxes = np.tile(boxes_tmp, (1, fuse_prob.shape[1]))
    
    '''
       Faster rcnn
    '''
    boxes_fast = rois[:, 1:5] / im_scales[0]
    scores_fast = np.reshape(scores_fast, [scores_fast.shape[0], -1])
    bbox_pred_fast = np.reshape(bbox_pred_fast, [bbox_pred_fast.shape[0], -1])
    
    if cfg.TEST.BBOX_REG:
    # Apply bounding-box regression deltas
      box_deltas = bbox_pred_fast
      pred_boxes_fast = bbox_transform_inv(torch.from_numpy(boxes_fast), torch.from_numpy(box_deltas)).numpy()
      pred_boxes_fast = _clip_boxes(pred_boxes_fast, im.shape)
    else:
    # Simply repeat the boxes, once for each class
      pred_boxes = np.tile(boxes_fast, (1, scores_fast.shape[1]))    
    
    
    
    cfg.TEST.USE_FLIPPED=True
    if cfg.TEST.USE_FLIPPED:
      blobs['data'][i:i+1] = blobs['data'][i:i+1][:, :, ::-1, :]
      width = blobs['data'][i:i+1].shape[2]
      oldx1 = blobs['boxes'][i][:, 1].copy()
      oldx2 = blobs['boxes'][i][:, 3].copy()
      blobs['boxes'][i][:, 1] = width - oldx2 - 1
      blobs['boxes'][i][:, 3] = width - oldx1 - 1
      assert (blobs['boxes'][i][:, 3] >= blobs['boxes'][i][:, 1]).all()
      
      cls_prob, bbox_prob, fuse_prob, image_prob, _ , _ , _= net.test_image(blobs['data'][i:i+1,:], blobs['im_info'], blobs['boxes'][i])
      scores_tmp += fuse_prob

    if cfg.DEDUP_BOXES > 0:
        # Map scores and predictions back to the original set of boxes
      scores_tmp = scores_tmp[inv_index, :]
      pred_boxes = pred_boxes[inv_index, :]

    if i == 0:        
       scores = np.copy(scores_tmp)
    else:
     scores += scores_tmp

  scores /= len(blobs['data']) * (1. + cfg.TEST.USE_FLIPPED)

  return scores, pred_boxes, scores_fast, pred_boxes_fast

def apply_nms(all_boxes, thresh):
  """Apply non-maximum suppression to all predicted boxes output by the
  test_net method.
  """
  num_classes = len(all_boxes)
  num_images = len(all_boxes[0])
  nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
  for cls_ind in range(num_classes):
    for im_ind in range(num_images):
      dets = all_boxes[cls_ind][im_ind]
      if dets == []:
        continue

      x1 = dets[:, 0]
      y1 = dets[:, 1]
      x2 = dets[:, 2]
      y2 = dets[:, 3]
      scores = dets[:, 4]
      inds = np.where((x2 > x1) & (y2 > y1))[0]
      dets = dets[inds,:]
      if dets == []:
        continue

      keep = nms(torch.from_numpy(dets), thresh).numpy()
      if len(keep) == 0:
        continue
      nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
  return nms_boxes

def test_train_net(net, imdb, weights_filename, max_per_image=100, thresh=0.):
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = len(imdb.image_index)
  # all detections are collected into:
  #  all_boxes[cls][image] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)
  all_boxes = [[[] for _ in range(num_images)]
         for _ in range(imdb.num_classes)]

  all_boxes_fast = [[[] for _ in range(num_images)]
         for _ in range(imdb.num_classes+1)]  

  output_dir = get_output_dir(imdb, weights_filename)  #voc_2007_test/default(tag)/vgg16_faster_rcnn_iter_15000

  # timers
  _t = {'im_detect' : Timer(), 'misc' : Timer()}

  roidb = imdb.roidb
    
    
  for i in range(num_images):
    im = cv2.imread(imdb.image_path_at(i))

    _t['im_detect'].tic()
    scores, boxes, scores_fast, boxes_fast = im_detect(net, im, roidb[i]['boxes'])
    _t['im_detect'].toc()

    _t['misc'].tic()


    for j in range(0, imdb.num_classes):
      inds = np.argmax(scores[:, j])
      all_boxes[j][i] = \
                np.hstack((boxes[inds, j*4:(j+1)*4].reshape(1, -1), 
                           np.array([[scores[inds, j]]])))
    '''
        start of faster part
    '''
    # skip j = 0, because it's the background class
    for j in range(1, imdb.num_classes+1):
      inds = np.argmax(scores_fast[:, j])
      all_boxes_fast[j][i] = \
                np.hstack((boxes_fast[inds, j*4:(j+1)*4].reshape(1, -1), 
                           np.array([[scores_fast[inds, j]]])))
    '''
        end of faster part
    '''


    _t['misc'].toc()

    print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
        .format(i + 1, num_images, _t['im_detect'].average_time(),
            _t['misc'].average_time()))


  output_dir_ws = output_dir + '/' + 'wsddn' 
  if not os.path.exists(output_dir_ws):
    os.makedirs(output_dir_ws)
  det_file = os.path.join(output_dir_ws, 'discovery.pkl')
  with open(det_file, 'wb') as f:
    pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  imdb.evaluate_discovery(all_boxes, output_dir_ws)



  all_boxes_fast = all_boxes_fast[1:]  # filter the background boxes  
  output_dir_fast = output_dir + '/' + 'faster'
  if not os.path.exists(output_dir_fast):
    os.makedirs(output_dir_fast)  
  det_file = os.path.join(output_dir_fast, 'discovery.pkl')
  with open(det_file, 'wb') as f:
    pickle.dump(all_boxes_fast, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  imdb.evaluate_discovery(all_boxes_fast, output_dir_fast)

