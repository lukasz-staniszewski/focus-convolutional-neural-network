<h1 align="center">Focus Convolutional Neural Network (Focus-CNN)</h1>
<h3 align="center">Łukasz Staniszewski</h2>
<h3 align="center">supervised by dr hab. Paweł Wawrzyński</h2>

<br>
<div align="center">
<img src="https://simargl.eu/images/partner-wut.png" alt="banner" width=700>
</div>

<h2 align="center"> I. Thesis abstract </h2>
The thesis presents a new solution to the problem of detecting objects in an image using neural networks, the Focus-CNN architecture that uses the attention mechanism to support its processing by applying translation, scaling and rotation operations to identify objects in the image with greater ease. The concept of the model was inspired by Faster R-CNN - the existing solution in this area and, likewise, is also a two-stage network, with each component learnable. The presented solution consists of three modules: the focus network (which plays the role of a proposal of regions with objects by providing parameters for image transformation), a transforming module (which transforms the input image based on parameters to obtain the image fragment that best indicates the object) and a classifying network (which decides the category of the object based on the clipped and twisted image part). The thesis goal was to implement a basic version of the model, and the result was code ensuring that each part of the architecture fulfills its purpose so that it can be developed in the future. The main conclusion of the experiments carried out on both images from the PASCAL VOC 2012 and COCO-2017 datasets, with the comparison of the network results with those of the Faster-RCNN model, is that the newly proposed solution achieves better performance than its competitor in this study and has the capacity to become one of the best detectors. In addition, it has been observed that the inclusion of rotation in the problem, although it does not guarantee better performance in locating objects in the image, can enhance architecture classification ability, which indicates the potential of this parameter in the solved task.

<h2 align="center"> II. Model overview </h2>

Focus Convolutional Neural Network (Focus-CNN) is a novel architecture for object detection in images with a new attention concept. It is based on three components:

1) the focus network that indicates in the input image a potential location where the object could be,
2) the transform module that transforms the whole image based on parameters provided by the focus network to create input for the classifier network,
3) the classifier network that verifies if the object is there.

Direct competitor for proposed network: <a href="https://arxiv.org/pdf/1506.01497">Faster R-CNN</a> with <a href="https://github.com/AlphaJia/pytorch-faster-rcnn">PyTorch implementation</a>.

<h2 align="center"> III. Architecture </h2>

### Focus Network

The focus network is fed with an image. It outputs 4 numbers:

1) $(x,y)$ coordinates of the location in the image where the object is likely to be,
2) $log(scale)$, where scale says how much the part of the object needs to be zoomed to the predefined resolution,
3) $\theta$ angle at which the indicated part needs to be rotated to its normal view,
4) $p$ likelihood at which the located part contains the wanted object, used for analysis purposes.

### Classifier

The classifier is fed with the zoomed and rotated part of the image indicated by the focus network. It outputs the class that says which object is there or “no object” if there is no object.

<h2 align="center"> IV. Training </h2>

### Focus network pretraining

The network is fed with original and rotated images from the dataset. Its job is to learn to indicate the smallest squares that contain the bounding boxes of the objects. The network is also fed with images that do not contain the object; its job then is to produce the likelihood value equal to zero (the rest of the outputs do not matter).

### Classifier pretraining

The network is pretraining with zoomed parts of the images from the dataset. These parts are either defined by the bounding boxes (then the output should be a correct class) or random parts (then the output should be “no”).

### Fine-tuning

The architecture is fed with images that either contain or do not contain the required objects. The classifier is to output a correct class or “no”. The gradient flows backwards through both networks.
The training can be based on a dataset of images with objects indicated by bounding boxes.

<h2 align="center"> V. Datasets </h2>

Experiments have been carried out on two benchmark datasets:

+ <a href="https://cocodataset.org/#home">PASCAL VOC 2012 TRAIN/VAL DATASET FOR OBJECT DETECTION</a>
+ <a href="http://host.robots.ox.ac.uk/pascal/VOC/">COCO 2017 TRAIN/VAL DATASET FOR OBJECT DETECTION</a>

<h2 align="center"> VI. Additional information </h2>

+ Trainings were performed using RTX4090, RTXA5000 and A100 GPUs.
+ In local development, Python 3.10 was used with venv, all necessary modules are in requirements.txt.
+ All experiments are reproducible thanks to random seeds.

<h2 align="center"> VII. Final results </h2>

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

