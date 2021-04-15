# -*- coding: utf-8 -*-
"""Training.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IBa_F2FaPwAOYj85SprKJDAyI8Dr_46J
"""

!pip uninstall tensorflow 
!pip install tensorflow-gpu==1.14

!pip uninstall keras 
!pip install keras==2.2.4

# Commented out IPython magic to ensure Python compatibility.
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline
#%load_ext line_profiler

import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.INFO)
sess_config = tf.ConfigProto()

import sys
import os

COCO_DATA = 'data/coco'
MASK_RCNN_MODEL_PATH = 'lib/Mask_RCNN/'

if MASK_RCNN_MODEL_PATH not in sys.path:
    sys.path.append(MASK_RCNN_MODEL_PATH)
    
from lib.Mask_RCNN.samples.coco import coco
from lib.Mask_RCNN.mrcnn import utils
from lib.Mask_RCNN.mrcnn import model as modellib
from lib.Mask_RCNN.mrcnn import visualize
    
from lib import utils as siamese_utils
from lib import model as siamese_model
from lib import config as siamese_config
   
import time
import datetime
import random
import numpy as np
import skimage.io
import imgaug
import pickle
import matplotlib.pyplot as plt
from collections import OrderedDict

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# train_classes = coco_nopascal_classes
train_classes = np.array(range(1,81))

# Load COCO/val dataset
coco_val = siamese_utils.IndexedCocoDataset()
coco_object = coco_val.load_coco(COCO_DATA, "val", year="2017",return_coco=True)
coco_object.getCatIds()
coco_val.prepare()
coco_val.build_indices()
coco_val.ACTIVE_CLASSES = train_classes

class SmallTrainConfig(siamese_config.Config):
    # Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 6 # A 16GB GPU is required for a batch_size of 12
    NUM_CLASSES = 1 + 1
    NAME = 'small_coco'
    EXPERIMENT = 'example'
    CHECKPOINT_DIR = 'checkpoints/'
    STEPS_PER_EPOCH = 4   # changed this to reduce load from 1000 to 4
    VALIDATION_STEPS = 1  # changed this from its og value in config
    # Adapt loss weights
    LOSS_WEIGHTS = {'rpn_class_loss': 2.0, 
                    'rpn_bbox_loss': 0.1, 
                    'mrcnn_class_loss': 2.0, 
                    'mrcnn_bbox_loss': 0.5, 
                    'mrcnn_mask_loss': 1.0}
    
class LargeTrainConfig(siamese_config.Config):
    # Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 4
    IMAGES_PER_GPU = 3 # 4 16GB GPUs are required for a batch_size of 12
    NUM_CLASSES = 1 + 1
    NAME = 'large_coco'
    EXPERIMENT = 'example'
    CHECKPOINT_DIR = 'checkpoints/'
    # Reduced image sizes
    TARGET_MAX_DIM = 192
    TARGET_MIN_DIM = 150
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    # Reduce model size
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024
    FPN_FEATUREMAPS = 256
    # Reduce number of rois at all stages
    RPN_ANCHOR_STRIDE = 1
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000
    TRAIN_ROIS_PER_IMAGE = 200
    DETECTION_MAX_INSTANCES = 100
    MAX_GT_INSTANCES = 100
    # Adapt NMS Threshold
    DETECTION_NMS_THRESHOLD = 0.5
    # Adapt loss weights
    LOSS_WEIGHTS = {'rpn_class_loss': 2.0, 
                    'rpn_bbox_loss': 0.1, 
                    'mrcnn_class_loss': 2.0, 
                    'mrcnn_bbox_loss': 0.5, 
                    'mrcnn_mask_loss': 1.0}

# The small model trains on a single GPU and runs much faster.
# The large model is the same we used in our experiments but needs multiple GPUs and more time for training.
model_size = 'small' # or 'large'

if model_size == 'small':
    config = SmallTrainConfig()
elif model_size == 'large':
    config = LargeTrainConfig()
    
config.display()

# Create model object in inference mode.
model = siamese_model.SiameseMaskRCNN(mode="training", model_dir=MODEL_DIR, config=config)

# train_schedule = OrderedDict()
# train_schedule[1] = {"learning_rate": config.LEARNING_RATE, "layers": "heads"}
# train_schedule[120] = {"learning_rate": config.LEARNING_RATE, "layers": "all"}
# train_schedule[160] = {"learning_rate": config.LEARNING_RATE/10, "layers": "all"}

# We commented out the above as google colab kept crashing, so we reduced the load
# Check if both folders under 'logs' folder are empty before running model.train
train_schedule = OrderedDict()
train_schedule[1] = {"learning_rate": config.LEARNING_RATE, "layers": "heads"}
train_schedule[3] = {"learning_rate": config.LEARNING_RATE, "layers": "all"}

# Load weights trained on Imagenet
try: 
    model.load_latest_checkpoint(training_schedule=train_schedule)
except:
    model.load_imagenet_weights(pretraining='imagenet-687')
l = []
for epochs, parameters in train_schedule.items():
    print("")
    print("training layers {} until epoch {} with learning_rate {}".format(parameters["layers"], 
                                                                          epochs, 
                                                                          parameters["learning_rate"]))
    history = model.train(coco_val, coco_val, 
                learning_rate=parameters["learning_rate"], 
                epochs=epochs, 
                layers=parameters["layers"])

print("Training completed")

## Plot the losses
plt.rcParams["figure.figsize"] = (7,5)
a= plt.figure()
plt.plot(range(1,epochs), history.history['val_loss'], 'r', label = 'Test Loss')
plt.plot(range(1,epochs), history.history['loss'], 'y', label = 'Train Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('/content/drive/MyDrive/siamese-mask-rcnn-master/train_plot.jpg')