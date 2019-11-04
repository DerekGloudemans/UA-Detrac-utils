"""
This file provides:
    - a Faster_RCNN model using the torchvision model package
    - training scripts for using the UA detrac data to train the model
    - simple evalatuation of results
No special modifications to the network or anything here, this is just a 
bare-bones attempt at using Faster-RCNN provided with Pytorch.
"""

# this seems to be a popular thing to do so I've done it here
from __future__ import print_function, division

# torch and specific torch packages for convenience
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.optim import lr_scheduler
from torch import multiprocessing
from torch.autograd import Variable

# for convenient data loading, image representation and dataset management
from torchvision import models, transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
from scipy.ndimage import affine_transform

# always good to have
import time
import os
import sys
import numpy as np    
import _pickle as pickle
import random
import copy
import matplotlib.pyplot as plt
import math

model = models.detection.fasterrcnn_resnet50_fpn(pretrained_backbone = True,num_classes = 8)
"""
The input to the model is expected to be a list of tensors, 
each of shape [C, H, W], one for each image, and should be in 0-1 range. 
Different images can have different sizes.

During training, the model expects both the input tensors, 
as well as a targets (list of dictionary), containing:

-boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, 
    with values between 0 and H and 0 and W
-labels (Int64Tensor[N]): the class label for each ground-truth box
-The model returns a Dict[Tensor] during training, containing the 
    classification and regression losses for both the RPN and the R-CNN.

During inference, the model requires only the input tensors, and returns the
post-processed predictions as a List[Dict[Tensor]], one for each input image. 
The fields of the Dict are as follows:

-boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, 
    with values between 0 and H and 0 and W
-labels (Int64Tensor[N]): the predicted labels for each image
-scores (Tensor[N]): the scores or each prediction
"""