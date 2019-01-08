from model.config import cfg
import numpy as np
import numpy.random as npr
from utils.bbox import bbox_overlaps
from model.bbox_transform import bbox_transform
from model.nms_wrapper import nms
import torch


def generate_pseudo_gtbox(boxes, cls_prob, im_labels):
    """Get proposals from fuse_matrix
    inputs are all variables"""
    pre_nms_topN = 50
    nms_Thresh = 0.1
    
    num_images, num_classes = im_labels.size()
    boxes = boxes[:,1:]
    assert num_images == 1, 'batch size shoud be equal to 1'
    im_labels_tmp = im_labels[0, :]
    labelList = im_labels_tmp.data.nonzero().view(-1)
    
    gt_boxes = []
    gt_classes = []
    gt_scores = []
    
    for i in labelList:
        scores, order = cls_prob[:,i].contiguous().view(-1).sort(descending=True)
        if pre_nms_topN > 0:
          order = order[:pre_nms_topN]
          scores = scores[:pre_nms_topN].view(-1, 1)
          proposals = boxes[order.data, :]
          
        keep = nms(torch.cat((proposals, scores), 1).data, nms_Thresh)
        proposals = proposals[keep, :]
        scores = scores[keep,]
        gt_boxes.append(proposals)
        gt_classes.append(torch.ones(keep.size(0),1)*(i+1))  # return idx=class+1 to include the background
        gt_scores.append(scores.view(-1,1))
            
    gt_boxes = torch.cat(gt_boxes)
    gt_classes = torch.cat(gt_classes)
    gt_scores = torch.cat(gt_scores)
    proposals = {'gt_boxes' : gt_boxes,
                 'gt_classes': gt_classes,
                 'gt_scores': gt_scores}
  #  print(gt_boxes.size())
  #  print(gt_classes.size())
  #  print(type(gt_boxes))
  #  print(type(gt_classes))
    return torch.cat([gt_boxes,gt_classes],1),proposals