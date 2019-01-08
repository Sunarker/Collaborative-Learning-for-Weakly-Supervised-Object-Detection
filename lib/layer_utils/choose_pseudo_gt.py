from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from model.config import cfg
import numpy as np
import numpy.random as npr
from utils.bbox import bbox_overlaps
from model.bbox_transform import bbox_transform
import torch


def choose_pseudo_gt(boxes, cls_prob, im_labels):
    """Get proposals with highest score.
    inputs are all variables"""

    num_images, num_classes = im_labels.size()
    boxes = boxes[:,1:]
    assert num_images == 1, 'batch size shoud be equal to 1'
    im_labels_tmp = im_labels[0, :]
    
    gt_boxes = []
    gt_classes = []
    gt_scores = []
    for i in range(num_classes):
        if im_labels_tmp[i].data.cpu().numpy() == 1:
            max_value,max_index = cls_prob[:, i].max(0)
            gt_boxes.append(boxes[max_index])
            gt_classes.append(torch.ones(1,1)*(i+1))  # return idx=class+1 to include the background
            gt_scores.append(max_value.view(-1,1))
            
    gt_boxes = torch.cat(gt_boxes)
    gt_classes = torch.cat(gt_classes)
    gt_scores = torch.cat(gt_scores)
    proposals = {'gt_boxes' : gt_boxes,
                 'gt_classes': gt_classes,
                 'gt_scores': gt_scores}
    
    return torch.cat([gt_boxes,gt_classes],1), proposals



