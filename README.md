<h1 align="center">Focus Convolutional Neural Network (Focus-CNN)</h1>
<h3 align="center">Łukasz Staniszewski</h2>
<h3 align="center">supervised by dr hab. Paweł Wawrzyński</h2>

<br>
<div align="center">
<img src="https://simargl.eu/images/partner-wut.png" alt="banner" width=700>
</div>

<h2 align="center"> I. Abstract </h2>

The proposed network Focus Convolutional Neural Network is an architecture for object detection in images. It is based on two neural component:

1) the focus network that indicates in the input image a potential location where the object could be
2) the classifier that verifies if the object is there.

Direct competitor for proposed network: <a href="https://arxiv.org/pdf/1506.01497">Faster R-CNN</a> with <a href="https://github.com/AlphaJia/pytorch-faster-rcnn">PyTorch implementation</a>.

<h2 align="center"> II. Architecture </h2>

### Focus Network

The focus network is fed with an image. It outputs 4 numbers:

1) $(x,y)$ coordinates of the location in the image where the object is likely to be,
2) $log(scale)$, where scale says how much the part of the object needs to be zoomed to the predefined resolution,
3) $\theta$ angle at which the indicated part needs to be rotated to its normal view,
4) $p$ likelihood at which the located part contains the wanted object.

### Classifier

The classifier is fed with the zoomed and rotated part of the image indicated by the focus network. It output one scalar that says if the image contains an object of the given class or not.

<h2 align="center"> III. Training </h2>

### Focus network pretraining

The network is fed with original and rotated images from the dataset. Its job is to learn to indicate the smallest squares that contain the bounding boxes of the objects. The network is also fed with the images that do not contain the object; its job then is to produce the likelihood value equal to zero (the rest of the outputs do not matter).

### Classifier pretraining

The network is pretraining with zoomed parts of the images from the dataset. These parts are either defined by the bounding boxes (then the output should be “yes”) or random parts (then the output should be “no”).

### Fine tuning

The architecture is fed with the images that either contain or not contain the required objects. The classifier is to output a correct “yes” or “no”. The gradient flows backward through both the networks.

The training can be based on a dataset of images with objects indicated by bounding boxes.

<h2 align="center"> IV. Datasets </h2>

Experiments will be carried out on two benchmark datasets:

+ <a href="https://cocodataset.org/#home">COCO 2017 TRAIN/VAL DATASET FOR OBJECT DETECTION</a>
+ TBA

<h2 align="center"> V. Additional information </h2>

+ Trainings are performed using Tesla T4 (Google Colab) and Nvidia RTX 2060M (locally) GPUs.
+ In local development, Python 3.9.5 was used with venv, all necessary modules are in requirements.txt.
+ All experiments are reproducible thanks to random seeds.
<!-- + Folder structure:
  ```
  footbal-frame-classifier/
  │
  ├── results.csv - final predictions
  ├── train.py - main script to start training
  ├── make_predictions.py - script for making predictions
  ├── preprocess_data.py - script for data preprocessing
  ├── evaluate_metrics_val.py - script for model validation  
  │
  ├── requirements.txt - necessary modules to develop locally
  ├── config.json - holds configuration for training
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── notebooks/ - notebooks used in project
  │   ├── DataAnalysis.ipynb - notebook for data preprocessing
  │   └── Colab.ipynb - Google Colab session using all scripts
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   └── base_trainer.py
  │
  ├── data_loader/ - anything about data loading goes here
  │   ├── TestDataset.py    - dataset for tests
  │   ├── FramesDataset.py  - dataset for train/validation
  │   └── data_loaders.py
  │
  ├── data/ - default directory for storing input data
  │
  ├── model/ - models, losses, and metrics defined
  │   ├── model.py
  │   ├── metric.py
  │   └── loss.py
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  ├── logger/ - module for tensorboard visualization and logging - not used in this project
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── ...
  ``` -->

<h2 align="center"> VI. Final metrics </h2>

### COCO 2017 DATASET

#### CLASSIFIER PRETRAINING

- Model path: /content/drive/MyDrive/GitHub/focus-convolutional-neural-network/res/classifiers/coco_classifier_multi/trainer/0627_075340/models//checkpoint-epoch73.pth ...
+ Testing model: CocoMultiClassifier
+ Predictions saved to res/classifiers/coco_classifier_multi/tester/0627_102942/predictions.csv
+ Metrics among all classes: {'micro_accuracy': 0.9673064947128296, 'micro_recall': 0.9346129894256592, 'micro_precision': 0.9346129894256592, 'micro_f1': 0.9346129894256592, 'macro_accuracy': 0.9673064947128296, 'macro_recall': 0.9022196531295776, 'macro_precision': 0.8579119443893433, 'macro_f1': 0.8785837888717651}
+ Metrics for each class:
┏━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
┃ Class      ┃ Accuracy ┃ Precision ┃ Recall ┃ F1-score ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━┩
│ none       │    0.938 │     0.969 │  0.932 │    0.950 │
│ person     │    0.954 │     0.902 │  0.949 │    0.925 │
│ car        │    0.981 │     0.775 │  0.895 │    0.831 │
│ bicycle    │    0.997 │     0.785 │  0.832 │    0.808 │
└────────────┴──────────┴───────────┴────────┴──────────┘

#### FOCUS NETWORK PRETRAINING

##### PERSON CLASS

##### CAR CLASS

##### BICYCLE CLASS

#### FOCUS-CNN

### PASCAL VOC DATASET

#### CLASSIFIER PRETRAINING

- Model path: /content/drive/MyDrive/GitHub/focus-convolutional-neural-network/res/classifiers/coco_classifier_multi/trainer/0627_075340/models//checkpoint-epoch73.pth ...
+ Testing model: CocoMultiClassifier
+ Predictions saved to res/classifiers/coco_classifier_multi/tester/0627_102942/predictions.csv
+ Metrics among all classes: {'micro_accuracy': 0.9673064947128296, 'micro_recall': 0.9346129894256592, 'micro_precision': 0.9346129894256592, 'micro_f1': 0.9346129894256592, 'macro_accuracy': 0.9673064947128296, 'macro_recall': 0.9022196531295776, 'macro_precision': 0.8579119443893433, 'macro_f1': 0.8785837888717651}
+ Metrics for each class:
┏━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
┃ Class      ┃ Accuracy ┃ Precision ┃ Recall ┃ F1-score ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━┩
│ none       │    0.938 │     0.969 │  0.932 │    0.950 │
│ person     │    0.954 │     0.902 │  0.949 │    0.925 │
│ car        │    0.981 │     0.775 │  0.895 │    0.831 │
│ bicycle    │    0.997 │     0.785 │  0.832 │    0.808 │
└────────────┴──────────┴───────────┴────────┴──────────┘

#### FOCUS NETWORK PRETRAINING

##### PERSON CLASS

##### CAR CLASS

##### BICYCLE CLASS
