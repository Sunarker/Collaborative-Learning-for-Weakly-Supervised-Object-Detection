# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# Modified by Jiajie Wang for ws-Faster-rcnn

# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import utils.timer

from layer_utils.snippets import generate_anchors_pre
from layer_utils.proposal_layer import proposal_layer
from layer_utils.proposal_top_layer import proposal_top_layer
from layer_utils.anchor_target_layer import anchor_target_layer
from layer_utils.proposal_target_layer import proposal_target_layer
from layer_utils.generate_pseudo_gtbox import generate_pseudo_gtbox
from layer_utils.loss_function import bootstrap_cross_entropy
from utils.visualization import draw_bounding_boxes

from layer_utils.roi_pooling.roi_pool import RoIPoolFunction

from model.config import cfg

import tensorboardX as tb

from scipy.misc import imresize

class Network(nn.Module):
  def __init__(self):
    nn.Module.__init__(self)
    self._predictions = {}
    self._losses = {}
    self._anchor_targets = {}
    self._proposal_targets = {}
    self._layers = {}
    self._gt_image = None
    self._act_summaries = {}
    self._score_summaries = {}
    self._event_summaries = {}
    self._image_gt_summaries = {}
    self._variables_to_fix = {}

  def _add_gt_image(self):
    # add back mean
    image = self._image_gt_summaries['image'] + cfg.PIXEL_MEANS
    image = imresize(image[0], self._im_info[:2] / self._im_info[2])
    # BGR to RGB (opencv uses BGR)
    self._gt_image = image[np.newaxis, :,:,::-1].copy(order='C')

  def _add_gt_image_summary(self):
    # use a customized visualization function to visualize the boxes
    self._add_gt_image()
    image = draw_bounding_boxes(\
                      self._gt_image, np.zeros((0,5)), self._image_gt_summaries['im_info'])  #no bounding_box ground_truth

    return tb.summary.image('GROUND_TRUTH', image[0].astype('float32')/255.0)

  def _add_act_summary(self, key, tensor):
    return tb.summary.histogram('ACT/' + key + '/activations', tensor.data.cpu().numpy(), bins='auto'),
    tb.summary.scalar('ACT/' + key + '/zero_fraction',
                      (tensor.data == 0).float().sum() / tensor.numel())

  def _add_score_summary(self, key, tensor):
    return tb.summary.histogram('SCORE/' + key + '/scores', tensor.data.cpu().numpy(), bins='auto')

  def _add_train_summary(self, key, var):
    return tb.summary.histogram('TRAIN/' + key, var.data.cpu().numpy(), bins='auto')

  def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred):
    rois, rpn_scores = proposal_top_layer(\
                                    rpn_cls_prob, rpn_bbox_pred, self._im_info,
                                     self._feat_stride, self._anchors, self._num_anchors)
    return rois, rpn_scores

  def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred):
    rois, rpn_scores = proposal_layer(\
                                    rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                                     self._feat_stride, self._anchors, self._num_anchors)

    return rois, rpn_scores


  def _roi_pool_layer(self, bottom, rois): # done
    return RoIPoolFunction(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1. / 16.)(bottom, rois)

  def _crop_pool_layer(self, bottom, rois, max_pool=True): # done
    # implement it using stn
    # box to affine
    # input (x1,y1,x2,y2)
    """
    [  x2-x1             x1 + x2 - W + 1  ]
    [  -----      0      ---------------  ]
    [  W - 1                  W - 1       ]
    [                                     ]
    [           y2-y1    y1 + y2 - H + 1  ]
    [    0      -----    ---------------  ]
    [           H - 1         H - 1      ]
    """
    rois = rois.detach()

    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = bottom.size(2)
    width = bottom.size(3)

    # affine theta
    theta = Variable(rois.data.new(rois.size(0), 2, 3).zero_())
    theta[:, 0, 0] = (x2 - x1) / (width - 1)
    theta[:, 0 ,2] = (x1 + x2 - width + 1) / (width - 1)
    theta[:, 1, 1] = (y2 - y1) / (height - 1)
    theta[:, 1, 2] = (y1 + y2 - height + 1) / (height - 1)

    if max_pool:
      pre_pool_size = cfg.POOLING_SIZE * 2
      grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, pre_pool_size, pre_pool_size)))
      crops = F.grid_sample(bottom.expand(rois.size(0), bottom.size(1), bottom.size(2), bottom.size(3)), grid)
      crops = F.max_pool2d(crops, 2, 2)
    else:
      grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, cfg.POOLING_SIZE, cfg.POOLING_SIZE)))
      crops = F.grid_sample(bottom.expand(rois.size(0), bottom.size(1), bottom.size(2), bottom.size(3)), grid)
    
    return crops


  def _anchor_target_layer(self, rpn_cls_score):
    rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, rpn_loss_weights = \
      anchor_target_layer(
      rpn_cls_score.data, self._gt_boxes.data.cpu().numpy(), self._pseudo_proposals['gt_scores'].data.cpu().numpy(), self._im_info, self._feat_stride, self._anchors.data.cpu().numpy(), self._num_anchors)

    rpn_labels = Variable(torch.from_numpy(rpn_labels).float().cuda()) #.set_shape([1, 1, None, None])
    rpn_bbox_targets = Variable(torch.from_numpy(rpn_bbox_targets).float().cuda())#.set_shape([1, None, None, self._num_anchors * 4])
    rpn_bbox_inside_weights = Variable(torch.from_numpy(rpn_bbox_inside_weights).float().cuda())#.set_shape([1, None, None, self._num_anchors * 4])
    rpn_bbox_outside_weights = Variable(torch.from_numpy(rpn_bbox_outside_weights).float().cuda())#.set_shape([1, None, None, self._num_anchors * 4])
    rpn_loss_weights = Variable(torch.from_numpy(rpn_loss_weights).float().cuda())#.set_shape([self._num_anchors])
    
    rpn_labels = rpn_labels.long()
    self._anchor_targets['rpn_labels'] = rpn_labels
    self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
    self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
    self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights
    self._anchor_targets['rpn_loss_weights'] = rpn_loss_weights
    
    for k in self._anchor_targets.keys():
      self._score_summaries[k] = self._anchor_targets[k]

    return rpn_labels

  def _proposal_target_layer(self, rois, roi_scores):
    rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, loss_weights = \
      proposal_target_layer(
      rois, roi_scores, self._gt_boxes, self._num_classes+1, self._pseudo_proposals['gt_scores'])

    self._proposal_targets['rois'] = rois
    self._proposal_targets['labels'] = labels.long()
    self._proposal_targets['bbox_targets'] = bbox_targets
    self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
    self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights
    self._proposal_targets['loss_weights'] = loss_weights
    for k in self._proposal_targets.keys():
      self._score_summaries[k] = self._proposal_targets[k]

    return rois, roi_scores

  def _anchor_component(self, height, width):
    # just to get the shape right
    #height = int(math.ceil(self._im_info.data[0, 0] / self._feat_stride[0]))
    #width = int(math.ceil(self._im_info.data[0, 1] / self._feat_stride[0]))
    anchors, anchor_length = generate_anchors_pre(\
                                          height, width,
                                           self._feat_stride, self._anchor_scales, self._anchor_ratios)
    self._anchors = Variable(torch.from_numpy(anchors).cuda())
    self._anchor_length = anchor_length

  def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
      loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box


  def _add_losses(self, sigma_rpn=3.0): 

    # classification loss
    image_prob = self._predictions["image_prob"]
    
#    assert ((image_prob.data>=0).sum()+(image_prob.data<=1).sum())==image_prob.data.size(1)*2, image_prob
#    assert ((self._labels.data>=0).sum()+(self._labels.data<=1).sum())==self._labels.data.size(1)*2, self._labels

    cross_entropy = F.binary_cross_entropy(image_prob.clamp(0,1),self._labels)
    
    fast_loss = self._add_losses_fast()
    self._losses['wsddn_loss'] = cross_entropy
    self._losses['fast_loss'] = fast_loss
    
    loss = cross_entropy + fast_loss
    self._losses['total_loss'] = loss
    
    for k in self._losses.keys():
      self._event_summaries[k] = self._losses[k]    
    return loss



  def _add_losses_fast(self, sigma_rpn=3.0):
    # RPN, class loss
    rpn_cls_score = self._predictions['rpn_cls_score_reshape'].view(-1, 2)
    rpn_label = self._anchor_targets['rpn_labels'].view(-1)
    rpn_loss_weights = self._anchor_targets['rpn_loss_weights']
    rpn_select = Variable((rpn_label.data != -1).nonzero().view(-1))
    rpn_cls_score = rpn_cls_score.index_select(0, rpn_select).contiguous().view(-1, 2)
    rpn_label = rpn_label.index_select(0, rpn_select).contiguous().view(-1)
    rpn_loss_weights = rpn_loss_weights.index_select(0, rpn_select).contiguous().view(-1)
    rpn_cross_entropy = bootstrap_cross_entropy(rpn_cls_score, rpn_label, ishard=cfg.TRAIN.ISHARD,  beta=cfg.TRAIN.BETA, weight=rpn_loss_weights)
#    rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label)
    
    # RPN, bbox loss
    rpn_bbox_pred = self._predictions['rpn_bbox_pred']
    rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
    rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
    rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']
    rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                          rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])

    # RCNN, class loss
    cls_score = self._predictions["cls_score_fast"]
    label = self._proposal_targets["labels"].view(-1)
    loss_weights = self._proposal_targets["loss_weights"].view(-1)
    cross_entropy = bootstrap_cross_entropy(cls_score.view(-1, self._num_classes+1), label, ishard=cfg.TRAIN.ISHARD,  beta=cfg.TRAIN.BETA, weight=loss_weights)
#    cross_entropy = F.cross_entropy(cls_score.view(-1, self._num_classes+1), label)
    
    # RCNN, bbox loss
    bbox_pred = self._predictions['bbox_pred_fast']
    bbox_targets = self._proposal_targets['bbox_targets']
    bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
    bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
    loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

    self._losses['cross_entropy_fast'] = cross_entropy
    self._losses['loss_box'] = loss_box
    self._losses['rpn_cross_entropy'] = rpn_cross_entropy
    self._losses['rpn_loss_box'] = rpn_loss_box

    loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box

    return loss



  def _region_proposal(self, net_conv):
    self._anchor_component(net_conv.size(2), net_conv.size(3))
    rpn = F.relu(self.rpn_net(net_conv))
    self._act_summaries['rpn'] = rpn

    rpn_cls_score = self.rpn_cls_score_net(rpn) # batch * (num_anchors * 2) * h * w

    # change it so that the score has 2 as its channel size
    rpn_cls_score_reshape = rpn_cls_score.view(1, 2, -1, rpn_cls_score.size()[-1]) # batch * 2 * (num_anchors*h) * w
    rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape)
    
    # Move channel to the last dimenstion, to fit the input of python functions
    rpn_cls_prob = rpn_cls_prob_reshape.view_as(rpn_cls_score).permute(0, 2, 3, 1) # batch * h * w * (num_anchors * 2)
    rpn_cls_score = rpn_cls_score.permute(0, 2, 3, 1) # batch * h * w * (num_anchors * 2)
    rpn_cls_score_reshape = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous()  # batch * (num_anchors*h) * w * 2
    rpn_cls_pred = torch.max(rpn_cls_score_reshape.view(-1, 2), 1)[1]

    rpn_bbox_pred = self.rpn_bbox_pred_net(rpn)
    rpn_bbox_pred = rpn_bbox_pred.permute(0, 2, 3, 1).contiguous()  # batch * h * w * (num_anchors*4)

    if self._mode == 'TRAIN':
    
      rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred) # rois, roi_scores are varible
      
      '''
         (3) do net_conv -> wsddn-branch  (TODO: put RPN-out into weights of WSDDN)
      '''
      fuse_prob = self._predict(net_conv)
      self._gt_boxes, self._pseudo_proposals  = self._generate_pseudo_gtbox(fuse_prob, self._boxes) # choose the `pseudo-gt boxes` to supervise faster-rcnn
      '''
         (_anchor_target_layer and _proposal_target_layer) need ```_gt_boxes``` to choose the rois for compute loss and choose rois for Fast-RCNN network
      '''
      rpn_labels = self._anchor_target_layer(rpn_cls_score)
      rois, _ = self._proposal_target_layer(rois, roi_scores)
    else:
      if cfg.TEST.MODE == 'nms':
        rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred)
      elif cfg.TEST.MODE == 'top':
        rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred)
      else:
        raise NotImplementedError

    self._predictions["rpn_cls_score"] = rpn_cls_score
    self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
    self._predictions["rpn_cls_prob"] = rpn_cls_prob
    self._predictions["rpn_cls_pred"] = rpn_cls_pred
    self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
    self._predictions["rois"] = rois

    return rois




  def _region_classification(self, fc7): 
    cls_score = self.cls_score_net(fc7)
    cls_pred = torch.max(cls_score, 1)[1]  # the prediction class of each bbox
    cls_prob = F.softmax(cls_score)
    bbox_pred = self.bbox_pred_net(fc7)
    bbox_prob = torch.stack([F.softmax(bbox_pred[:,i]) for i in range(bbox_pred.size(1))], 1)
    fuse_prob = cls_prob.mul(bbox_prob)
    image_prob = fuse_prob.sum(0,keepdim=True)
    
    self._predictions["cls_pred"] = cls_pred
    self._predictions["cls_prob"] = cls_prob
    self._predictions["bbox_prob"] = bbox_prob
    self._predictions["fuse_prob"] = fuse_prob
    self._predictions["image_prob"] = image_prob

    return cls_prob, bbox_prob, fuse_prob, image_prob

  def _region_classification_fast(self, fc7):
    cls_score = self.cls_score_net_fast(fc7)
    cls_pred = torch.max(cls_score, 1)[1]
    cls_prob = F.softmax(cls_score)
    bbox_pred = self.bbox_pred_net_fast(fc7)

    self._predictions["cls_score_fast"] = cls_score
    self._predictions["cls_pred_fast"] = cls_pred
    self._predictions["cls_prob_fast"] = cls_prob
    self._predictions["bbox_pred_fast"] = bbox_pred

    return cls_prob, bbox_pred




  def _image_to_head(self):
    raise NotImplementedError

  def _head_to_tail(self, pool5):
    raise NotImplementedError

  def create_architecture(self, num_classes, tag=None,
                          anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)): #done
    self._tag = tag

    self._num_classes = num_classes

    self._anchor_scales = anchor_scales
    self._num_scales = len(anchor_scales)

    self._anchor_ratios = anchor_ratios
    self._num_ratios = len(anchor_ratios)

    self._num_anchors = self._num_scales * self._num_ratios

    assert tag != None

    # Initialize layers
    self._init_modules()

  def _init_modules(self): 
    self._init_head_tail()

    # rpn
    self.rpn_net = nn.Conv2d(self._net_conv_channels, cfg.RPN_CHANNELS, [3, 3], padding=1)

    self.rpn_cls_score_net = nn.Conv2d(cfg.RPN_CHANNELS, self._num_anchors * 2, [1, 1])
    
    self.rpn_bbox_pred_net = nn.Conv2d(cfg.RPN_CHANNELS, self._num_anchors * 4, [1, 1])

    self.cls_score_net_fast = nn.Linear(self._fc7_channels, self._num_classes+1)
    self.bbox_pred_net_fast = nn.Linear(self._fc7_channels, (self._num_classes+1) * 4)


    self.cls_score_net = nn.Linear(self._fc7_channels, self._num_classes)  # between class
    self.bbox_pred_net = nn.Linear(self._fc7_channels, self._num_classes)  # between boxes

    self.init_weights()

  def _run_summary_op(self, val=False):  
    """
    Run the summary operator: feed the placeholders with corresponding newtork outputs(activations)
    """
    summaries = []
    # Add image gt
    # summaries.append(self._add_gt_image_summary())
    # Add event_summaries
    for key, var in self._event_summaries.items():
      summaries.append(tb.summary.scalar(key, var.data[0]))
    self._event_summaries = {}
    #if not val:
      # Add score summaries
    #  for key, var in self._score_summaries.items():
    #    summaries.append(self._add_score_summary(key, var))
    #  self._score_summaries = {}
      # Add act summaries
      #for key, var in self._act_summaries.items():
      #  summaries += self._add_act_summary(key, var)
      #self._act_summaries = {}
      # Add train summaries
   #   for k, var in dict(self.named_parameters()).items():
   #     if var.requires_grad:
   #       summaries.append(self._add_train_summary(k, var))

   #   self._image_gt_summaries = {}
    
    return summaries

  def _generate_pseudo_gtbox(self, fuse_prob, boxes): # Inputs are two variables
      #return gt_boxes Variable(torch.from_numpy(gt_boxes).cuda()) size: gt_num * (x1,y1,x2,y2,class)
    gt_boxes, proposals = generate_pseudo_gtbox(boxes, fuse_prob, self._labels)
    return gt_boxes, proposals


  def _predict(self, net_conv):  
    # This is just _build_network in tf-faster-rcnn
   # torch.backends.cudnn.benchmark = False
   # net_conv = self._image_to_head()
    
    
    '''
        ROI pooling on SELECTIVE SEARCH boxes
    '''
    if cfg.POOLING_MODE == 'crop':
      pool5 = self._crop_pool_layer(net_conv, self._boxes)
    else:
      pool5 = self._roi_pool_layer(net_conv, self._boxes)

    if self._mode == 'TRAIN':
      torch.backends.cudnn.benchmark = True # benchmark because now the input size are fixed
    fc7 = self._head_to_tail(pool5)

    cls_prob, bbox_prob, fuse_prob, image_prob = self._region_classification(fc7)

#    for k in self._predictions.keys():
#      self._score_summaries[k] = self._predictions[k]
    self._score_summaries['image_prob'] = self._predictions['image_prob']
    #print(id(net_conv))
    return fuse_prob
#    return net_conv, cls_prob, bbox_prob, fuse_prob, image_prob
    

  def _predict_fast(self, net_conv):
    rois = self._region_proposal(net_conv)
    pool5_fast = self._roi_pool_layer(net_conv, rois)
    fc7_fast = self._head_to_tail(pool5_fast)
    cls_prob_fast, bbox_pred_fast = self._region_classification_fast(fc7_fast)    
    
#    for k in self._predictions.keys():
#      self._score_summaries[k] = self._predictions[k]
    return cls_prob_fast, bbox_pred_fast

  def forward(self, image, im_info, boxes, labels=None, mode='TRAIN'): #done
    self._image_gt_summaries['image'] = image
    self._image_gt_summaries['boxes'] = boxes
    self._image_gt_summaries['im_info'] = im_info
    self._image_gt_summaries['labels'] = labels
    
    self._image = Variable(torch.from_numpy(image.transpose([0,3,1,2])).cuda(), volatile=mode == 'TEST')
    self._im_info = im_info # No need to change; actually it can be an list
    self._boxes = Variable(torch.from_numpy(boxes).type('torch.Tensor').cuda())
    self._labels = Variable(torch.from_numpy(labels).type('torch.Tensor').cuda()) if labels is not None else None


    self._mode = mode
    '''
       (1) do image -> net_conv 
    '''
    torch.backends.cudnn.benchmark = False
    net_conv = self._image_to_head()

    '''
       (2) do net_conv -> faster-branch 
    '''
    cls_prob_fast, bbox_pred_fast = self._predict_fast(net_conv)
    
    

    if mode == 'TEST':
      if 1:
          '''
              (3) do net_conv -> wsddn-branch 
          '''
          fuse_prob = self._predict(net_conv)
      stds = bbox_pred_fast.data.new(cfg.TRAIN.BBOX_NORMALIZE_STDS).repeat(self._num_classes + 1).unsqueeze(0).expand_as(bbox_pred_fast)
      means = bbox_pred_fast.data.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS).repeat(self._num_classes + 1).unsqueeze(0).expand_as(bbox_pred_fast)
      self._predictions["bbox_pred_fast"] = bbox_pred_fast.mul(Variable(stds)).add(Variable(means))
    else:
      self._add_losses() # compute losses

  def init_weights(self): 
    def normal_init(m, mean, stddev, truncated=False):
      """
      weight initalizer: truncated normal and random normal.
      """
      # x is a parameter
      if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
      else:
        m.weight.data.normal_(mean, stddev)
      m.bias.data.zero_()
      
    normal_init(self.cls_score_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.bbox_pred_net, 0, 0.001, cfg.TRAIN.TRUNCATED)
    normal_init(self.rpn_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.rpn_cls_score_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.rpn_bbox_pred_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.cls_score_net_fast, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.bbox_pred_net_fast, 0, 0.001, cfg.TRAIN.TRUNCATED)
  # Extract the head feature maps, for example for vgg16 it is conv5_3
  # only useful during testing mode
  def extract_head(self, image): 
    feat = self._layers["head"](Variable(torch.from_numpy(image.transpose([0,3,1,2])).cuda(), volatile=True))
    return feat

  # only useful during testing mode
  def test_image(self, image, im_info, boxes): 
    self.eval()
    self.forward(image, im_info, boxes, None, mode='TEST')
    cls_prob, bbox_prob, fuse_prob, image_prob = self._predictions["cls_prob"].data.cpu().numpy(), \
                                                     self._predictions['bbox_prob'].data.cpu().numpy(), \
                                                     self._predictions['fuse_prob'].data.cpu().numpy(), \
                                                     self._predictions['image_prob'].data.cpu().numpy()
    
    cls_prob_fast, bbox_pred_fast, rois = self._predictions['cls_prob_fast'].data.cpu().numpy(), \
                                                     self._predictions['bbox_pred_fast'].data.cpu().numpy(), \
                                                     self._predictions['rois'].data.cpu().numpy()
    
    return cls_prob, bbox_prob, fuse_prob, image_prob, cls_prob_fast, bbox_pred_fast, rois

  def delete_intermediate_states(self): 
    # Delete intermediate result to save memory
    for d in [self._losses, self._predictions, self._anchor_targets, self._proposal_targets]:
      for k in list(d):
        del d[k]

  def get_summary(self, blobs): 
    self.eval()
    self.forward(blobs['data'], blobs['im_info'], blobs['boxes'], blobs['labels'])
    self.train()
    summary = self._run_summary_op(True)

    return summary

  def train_step(self, blobs, train_op):  
    self.forward(blobs['data'], blobs['im_info'], blobs['boxes'], blobs['labels'])
    cross_entropy, total_loss = self._losses['wsddn_loss'].data[0], \
                          self._losses['total_loss'].data[0]
    #utils.timer.timer.tic('backward')
    train_op.zero_grad()
    self._losses['total_loss'].backward()
    #utils.timer.timer.toc('backward')
    train_op.step()

    self.delete_intermediate_states()

    return cross_entropy, total_loss

  def train_step_with_summary(self, blobs, train_op): 
    self.forward(blobs['data'], blobs['im_info'], blobs['boxes'], blobs['labels'])
    cross_entropy, total_loss = self._losses['wsddn_loss'].data[0], \
                          self._losses['total_loss'].data[0]
                          
    train_op.zero_grad()
    self._losses['total_loss'].backward()
    train_op.step()
    summary = self._run_summary_op()

    self.delete_intermediate_states()

    return cross_entropy, total_loss, summary

  def train_step_no_return(self, blobs, train_op):  
    self.forward(blobs['data'], blobs['im_info'], blobs['boxes'], blobs['labels'])
    train_op.zero_grad()
    self._losses['total_loss'].backward()
    train_op.step()
    self.delete_intermediate_states()

  def load_state_dict(self, state_dict):
    """
    Because we remove the definition of fc layer in resnet now, it will fail when loading 
    the model trained before.
    To provide back compatibility, we overwrite the load_state_dict
    """
    nn.Module.load_state_dict(self, {k: state_dict[k] for k in list(self.state_dict())})

