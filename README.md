# Collaborative Learning for Weakly Supervised Object Detection

If you use this code in your research, please cite
```
@inproceedings{ijcai2018-135,
  title     = {Collaborative Learning for Weakly Supervised Object Detection},
  author    = {Jiajie Wang and Jiangchao Yao and Ya Zhang and Rui Zhang},
  booktitle = {Proceedings of the Twenty-Seventh International Joint Conference on
               Artificial Intelligence, {IJCAI-18}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  pages     = {971--977},
  year      = {2018},
  month     = {7},
  doi       = {10.24963/ijcai.2018/135},
  url       = {https://doi.org/10.24963/ijcai.2018/135},
}
```

# Important notice:
If you used the master branch before Sep. 26 2017 and its corresponding pretrained model, **PLEASE PAY ATTENTION**:
The old master branch in now under old_master, you can still run the code and download the pretrained model, but the pretrained model for that old master is not compatible to the current master!

The main differences between new and old master branch are in this two commits: [9d4c24e](https://github.com/ruotianluo/pytorch-faster-rcnn/commit/9d4c24e83c3e4ec33751e50d5e4d8b1dd793dfaa), [c899ce7](https://github.com/ruotianluo/pytorch-faster-rcnn/commit/c899ce70dae62e3db1a5805eda96df88e4b59ca6)
The change is related to this [issue](https://github.com/ruotianluo/pytorch-faster-rcnn/issues/6); master now matches all the details in [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn) so that we can now convert pretrained tf model to pytorch model.

# pytorch-faster-rcnn
A pytorch implementation of faster RCNN detection framework based on Xinlei Chen's [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn). Xinlei Chen's repository is based on the python Caffe implementation of faster RCNN available [here](https://github.com/rbgirshick/py-faster-rcnn).

**Note**: Several minor modifications are made when reimplementing the framework, which give potential improvements. For details about the modifications and ablative analysis, please refer to the technical report [An Implementation of Faster RCNN with Study for Region Sampling](https://arxiv.org/pdf/1702.02138.pdf). If you are seeking to reproduce the results in the original paper, please use the [official code](https://github.com/ShaoqingRen/faster_rcnn) or maybe the [semi-official code](https://github.com/rbgirshick/py-faster-rcnn). For details about the faster RCNN architecture please refer to the paper [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/pdf/1506.01497.pdf).

### Prerequisites
  - A basic pytorch installation. The code follows **0.2**. If you are using old **0.1.12**, you can checkout 0.1.12 branch.
  - Python packages you might not have: `cffi`, `opencv-python`, `easydict` (similar to [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)). For `easydict` make sure you have the right version. Xinlei uses 1.6.
  - [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch) to visualize the training and validation curve. Please build from source to use the latest tensorflow-tensorboard.
  - ~~Docker users: Since the recent upgrade, the docker image on docker hub (https://hub.docker.com/r/mbuckler/tf-faster-rcnn-deps/) is no longer valid. However, you can still build your own image by using dockerfile located at `docker` folder (cuda 8 version, as it is required by Tensorflow r1.0.) And make sure following Tensorflow installation to install and use nvidia-docker[https://github.com/NVIDIA/nvidia-docker]. Last, after launching the container, you have to build the Cython modules within the running container.~~

### Installation
1. Clone the repository
  ```Shell
  git clone https://github.com/ruotianluo/pytorch-faster-rcnn.git
  ```

2. Choose your `-arch` option to match your GPU for step 3 and 4.

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |

  **Note**: You are welcome to contribute the settings on your end if you have made the code work properly on other GPUs.


3. Build RoiPooling module
  ```
  cd pytorch-faster-rcnn/lib/layer_utils/roi_pooling/src/cuda
  echo "Compiling roi_pooling kernels by nvcc..."
  nvcc -c -o roi_pooling_kernel.cu.o roi_pooling_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
  cd ../../
  python build.py
  cd ../../../
  ```


4. Build NMS
  ```
  cd lib/nms/src/cuda
  echo "Compiling nms kernels by nvcc..."
  nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
  cd ../../
  python build.py
  cd ../../
  ```


### Setup data
Please follow the instructions of py-faster-rcnn [here](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to setup VOC. The steps involve downloading data and optionally creating soft links in the ``data`` folder. Since faster RCNN does not rely on pre-computed proposals, it is safe to ignore the steps that setup proposals.

If you find it useful, the ``data/cache`` folder created on Xinlei's side is also shared [here](http://ladoga.graphics.cs.cmu.edu/xinleic/tf-faster-rcnn/cache.tgz).


### Train your own model
1. Download pre-trained models and weights. The current code support VGG16 and Resnet V1 models. Pre-trained models are provided by [pytorch-vgg](https://github.com/jcjohnson/pytorch-vgg.git) and [pytorch-resnet](https://github.com/ruotianluo/pytorch-resnet) (the ones with caffe in the name), you can download the pre-trained models and set them in the ``data/imagenet_weights`` folder. For example for VGG16 model, you can set up like:
   ```Shell
   mkdir -p data/imagenet_weights
   cd data/imagenet_weights
   python # open python in terminal and run the following Python code
   ```
   ```Python
   import torch
   from torch.utils.model_zoo import load_url
   from torchvision import models

   sd = load_url("https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth")
   sd['classifier.0.weight'] = sd['classifier.1.weight']
   sd['classifier.0.bias'] = sd['classifier.1.bias']
   del sd['classifier.1.weight']
   del sd['classifier.1.bias']

   sd['classifier.3.weight'] = sd['classifier.4.weight']
   sd['classifier.3.bias'] = sd['classifier.4.bias']
   del sd['classifier.4.weight']
   del sd['classifier.4.bias']

   torch.save(sd, "vgg16.pth")
   ```
   ```Shell
   cd ../..
   ```
   For Resnet101, you can set up like:
   ```Shell
   mkdir -p data/imagenet_weights
   cd data/imagenet_weights
   # download from my gdrive (link in pytorch-resnet)
   mv resnet101-caffe.pth res101.pth
   cd ../..
   ```

2. Train (and test, evaluation)
  ```Shell
  ./experiments/scripts/train.sh [GPU_ID] [DATASET] [NET] [WSDDN_PRETRAINED]
  # Examples:
  ./experiments/scripts/train.sh 0 pascal_voc vgg16 path_to_wsddn_pretrained_model
  ```

3. Visualization with Tensorboard
  ```Shell
  tensorboard --logdir=tensorboard/vgg16/voc_2007_trainval/ --port=7001 &
  ```

4. Test and evaluate
  ```Shell
  ./experiments/scripts/test.sh [GPU_ID] [DATASET] [NET] [WSDDN_PRETRAINED]
  # Examples:
  ./experiments/scripts/test.sh 0 pascal_voc vgg16 path_to_wsddn_pretrained_model
  ```



By default, trained networks are saved under:

```
output/[NET]/[DATASET]/default/
```

Test outputs are saved under:

```
output/[NET]/[DATASET]/default/[SNAPSHOT]/
```

Tensorboard information for train and validation is saved under:

```
tensorboard/[NET]/[DATASET]/default/
tensorboard/[NET]/[DATASET]/default_val/
```
