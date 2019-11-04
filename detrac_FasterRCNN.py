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

from detrac_dataset import Track_Dataset


class ListBatch:
    def __init__(self, data):
        self.transposed_data = list(zip(*data))
        self.transposed_data[0] = torch.stack(self.transposed_data[0],dim = 0)
        self.transposed_data[1] = list(self.transposed_data[1])
        
def collate_wrapper(batch):
    return ListBatch(batch).transposed_data

def train_model(model, optimizer, scheduler, 
                    dataloaders,dataset_sizes, num_epochs=5, start_epoch = 0):
        """
        Alternates between a training step and a validation step at each epoch. 
        Validation results are reported but don't impact model weights
        Trains RPN on even epochs, ROI heads on odd epochs
        """
        for epoch in range(start_epoch,num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)
    
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    if epoch > 0:
                        scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode
    
                # Iterate over data.
                count = 0
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()
    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        losses = model(inputs,labels)


                        if phase == 'train' :
                            if epoch %2 == 1:
                                losses['loss_classifier'].backward(retain_graph = True)
                                losses['loss_box_reg'].backward(retain_graph = False)
                            else:
                                losses['loss_objectness'].backward(retain_graph = True)
                                losses['loss_rpn_box_reg'].backward(retain_graph = False)
                            optimizer.step()
        
                    # verbose update
                    count += 1
                    if count % 1 == 0:
                        outstrings = ["{}:{}".format(item,losses[item]) for item in losses]
                        #print("on minibatch {} -- correct: {} -- avg bbox iou: {} ".format(count,correct,bbox_acc))
                        print(outstrings)
                
                torch.cuda.empty_cache()
                
        return model




if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')    
    except:
        pass
    
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.cuda.empty_cache()   

    #%% Create Model
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

    #%% Create dataloader and dataset
    
    label_dir = "C:\\Users\\derek\\Desktop\\UA Detrac\\DETRAC-Train-Annotations-XML-v3"
    image_dir = "C:\\Users\\derek\\Desktop\\UA Detrac\\Tracks"
    train_dataset = Track_Dataset(image_dir,label_dir,mode = "training")
    test_dataset = Track_Dataset(image_dir,label_dir,mode = "testing")
    
    # create training params
    params = {'batch_size': 2,
              'shuffle': True,
              'num_workers': 0,
              'collate_fn':collate_wrapper}
    trainloader = data.DataLoader(train_dataset,**params)
    testloader = data.DataLoader(test_dataset,**params)
    
#    start = time.time()
#    X, Y = next(iter(trainloader))
#    elapsed = time.time() - start
#    
#    X = X.to(device)
#    model = model.to(device)
#
#    out = model(X,Y)
    
    
    
    
    # all parameters are being optimized, not just fc layer
    #optimizer = optim.Adam(model.parameters(), lr=0.01)
    optimizer = optim.SGD(model.parameters(), lr=0.01,momentum = 0.9)    
    # Decay LR by a factor of 0.5 every epoch
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    start_epoch = 0
    num_epochs = 10

#    # if checkpoint specified, load model and optimizer weights from checkpoint
#    if checkpoint_file != None:
#        model,optimizer,start_epoch = load_model(checkpoint_file, model, optimizer)
#        #model,_,_ = load_model(checkpoint_file, model, optimizer) # optimizer restarts from scratch
#        print("Checkpoint loaded.")
            
    # group dataloaders
    dataloaders = {"train":trainloader, "val": testloader}
    datasizes = {"train": len(train_dataset), "val": len(test_dataset)}
    
    
    if True:    
    # train model
        print("Beginning training on {}.".format(device))
        model = train_model(model,  optimizer, 
                            exp_lr_scheduler, dataloaders,datasizes,
                            num_epochs, start_epoch)
    