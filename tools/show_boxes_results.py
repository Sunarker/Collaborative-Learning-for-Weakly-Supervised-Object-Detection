# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 01:50:15 2017

@author: jjwang
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import _init_paths
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
from datasets.factory import get_imdb
import datasets.imdb
import argparse
import pprint
import numpy as np
import sys
import os 
import pickle as pickle
import cv2
from matplotlib import pyplot as plt


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='show the imgs and the resulted boxes')
  parser.add_argument('--box', default='/DATA3_DB7/data/jjwang/workspace/wsFaster-rcnn/output/vgg16/voc_2007_test/WSDDN_PRE_50000/vgg16_faster_rcnn_iter_90000/wsddn/detections.pkl', help='boxes pkl file to load')
  parser.add_argument('--thr', default=0.1, type=float, help='idx of test img')

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args





if __name__ == '__main__':
  args = parse_args()
  print('Called with args:')
  print(args)
  
  
  
  with open(args.box, 'rb') as fid:
    try:
         content = pickle.load(fid)
    except:
         content = pickle.load(fid, encoding='bytes')
   

  boxpathList = args.box.split('/')
  save_base = '/'.join(boxpathList[-5:-1])
  save_path = os.path.join('../cache',save_base)
  save_path = os.path.join(save_path, boxpathList[-1].split('.')[0])
  if not os.path.exists(save_path):
     os.makedirs(save_path)
  save_path = '../cache/' + save_path 
  imdbname = boxpathList[-5]
  print('getting imdb {:s}'.format(imdbname))
  imdb = get_imdb('voc_2007_test')
  
  for idx in range(len(imdb.image_index)):
        im = cv2.imread(imdb.image_path_at(idx))
        im = im[:,:,::-1]
        height, width, depth = im.shape
        dpi = 80
        plt.figure(figsize=(width/dpi,height/dpi),dpi=dpi)
        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
        plt.imshow(im)  # plot the image for matplotlib
        currentAxis = plt.gca()
        plt.axis('off')
        # scale each detection back up to the image
        # scale = torch.Tensor([rgb_image.shape[1::-1], rgb_image.shape[1::-1]])
        for i in range(20):
            for j in range(len(content[i][idx])):
                score = content[i][idx][j][-1]
                if score > 0.1:
                    label_name = imdb._classes[i]
                    display_txt = '%s: %.2f'%(label_name, score)
                    pt = content[i][idx][j][:-1]
                    coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
                    color = colors[i]
                    currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
                    currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
        
        plt.savefig(save_path + '/' + imdb.image_index[idx] + '.jpg')
        plt.close() 
        if idx % 500 == 0 :
            print(idx)
      
      
  
  