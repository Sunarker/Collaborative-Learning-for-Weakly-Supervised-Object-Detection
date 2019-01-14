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
1. Download pre-trained models and weights. For the pretrained [wsddn](https://www.robots.ox.ac.uk/~vgg/publications/2016/Bilen16/bilen16.pdf) model, you can find the download link [here](https://github.com/hbilen/WSDDN). For other pre-trained models like VGG16 and Resnet V1 models, they are provided by [pytorch-vgg](https://github.com/jcjohnson/pytorch-vgg.git) and [pytorch-resnet](https://github.com/ruotianluo/pytorch-resnet) (the ones with caffe in the name). You can download the pre-trained models and set them in the ``data/imagenet_weights`` folder. For example for VGG16 model, you can set up like:
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
