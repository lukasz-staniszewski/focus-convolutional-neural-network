<h1 align="center">Focus Convolutional Neural Network (Focus-CNN)</h1>
<h3 align="center">Łukasz Staniszewski</h2>
<h3 align="center">supervised by dr hab. Paweł Wawrzyński</h2>

<br>
<div align="center">
<img src="https://simargl.eu/images/partner-wut.png" alt="banner" width=700>
</div>

<h2 align="center"> I. Abstract </h2>

Focus Convolutional Neural Network (Focus-CNN) is a novel architecture for object detection in images with a new attention concept. It is based on three components:

1) the focus network that indicates in the input image a potential location where the object could be,
2) the transform module that transforms whole image based on parameters provided by the focus network to create input for classifier network,
3) the classifier network that verifies if the object is there.

Direct competitor for proposed network: <a href="https://arxiv.org/pdf/1506.01497">Faster R-CNN</a> with <a href="https://github.com/AlphaJia/pytorch-faster-rcnn">PyTorch implementation</a>.

<h2 align="center"> II. Architecture </h2>

### Focus Network

The focus network is fed with an image. It outputs 4 numbers:

1) $(x,y)$ coordinates of the location in the image where the object is likely to be,
2) $log(scale)$, where scale says how much the part of the object needs to be zoomed to the predefined resolution,
3) $\theta$ angle at which the indicated part needs to be rotated to its normal view,
4) $p$ likelihood at which the located part contains the wanted object, used for analysis purpose.

### Classifier

The classifier is fed with the zoomed and rotated part of the image indicated by the focus network. It output the class that says which object is there or “no object” if there is no object.

<h2 align="center"> III. Training </h2>

### Focus network pretraining

The network is fed with original and rotated images from the dataset. Its job is to learn to indicate the smallest squares that contain the bounding boxes of the objects. The network is also fed with the images that do not contain the object; its job then is to produce the likelihood value equal to zero (the rest of the outputs do not matter).

### Classifier pretraining

The network is pretraining with zoomed parts of the images from the dataset. These parts are either defined by the bounding boxes (then the output should be a correct class) or random parts (then the output should be “no”).

### Fine tuning

The architecture is fed with the images that either contain or not contain the required objects. The classifier is to output a correct class or “no”. The gradient flows backward through both the networks.
The training can be based on a dataset of images with objects indicated by bounding boxes.

<h2 align="center"> IV. Datasets </h2>

Experiments have been carried out on two benchmark datasets:

+ <a href="https://cocodataset.org/#home">PASCAL VOC 2012 TRAIN/VAL DATASET FOR OBJECT DETECTION</a>
+ <a href="http://host.robots.ox.ac.uk/pascal/VOC/">COCO 2017 TRAIN/VAL DATASET FOR OBJECT DETECTION</a>

<h2 align="center"> V. Additional information </h2>

+ Trainings were performed using RTX4090, RTXA5000 and A100 GPUs.
+ In local development, Python 3.10 was used with venv, all necessary modules are in requirements.txt.
+ All experiments are reproducible thanks to random seeds.

<h2 align="center"> VI. Final results </h2>

### FASTER R-CNN

| **Dataset** | **$mAP@0.5$** | **$mAP@0.5:0.05:0.95$** |
|---|---|---|
| PASCAL VOC 2012 | 0,420 | 0,244 |
| MS COCO 2017 | 0,277 | 0,169 |

### FOCUS-CNN

| **Dataset** | **$mAP@0.5$** | **$mAP@0.5:0.05:0.95$** |
|---|---|---|
| PASCAL VOC 2012 | 0.545 | 0.324 |
| MS COCO 2017 | 0.499 | 0.301 |

### FOCUS-CNN WITH ROTATION

| **Dataset** | **$mAP@0.5$** | **$mAP@0.5:0.05:0.95$** |
|---|---|---|
| PASCAL VOC 2012 | 0.362 | 0.106 |

