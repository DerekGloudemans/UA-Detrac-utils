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
        best_loss = np.inf
        for epoch in range(start_epoch,num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)
    
            running_loss = 0
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    if epoch > 0:
                        scheduler.step()
                    model.train()  # Set model to training mode
#                else:
#                    model.eval()   # Set model to evaluate mode
    
                # Iterate over data.
                count = 0
                total_num_minibatches = dataset_sizes[phase] / dataloaders[phase].batch_size
                
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    device_labels = []
                    for label in labels:
                        label['boxes'] = label['boxes'].to(device)
                        label['labels'] = label['labels'].to(device)
                        device_labels.append(label)
                    # zero the parameter gradients
                    optimizer.zero_grad()
    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        losses = model(inputs,device_labels)


                        if phase == 'train' :
                            # deal with nan and inf values
                            bad_losses = []
                            for item in losses:
                                if torch.isnan(losses[item]) or losses[item] > 100:
                                    bad_losses.append(item)
                            for item in bad_losses:
                                del losses[item]
                                    
#                            if count//500 % 2 == 1:#epoch %2 == 1: # switch every 500 batches
#                                losses['loss_classifier'].backward(retain_graph = True)
#                                losses['loss_box_reg'].backward(retain_graph = False)
#                            else:
#                                losses['loss_objectness'].backward(retain_graph = True)
#                                losses['loss_rpn_box_reg'].backward(retain_graph = False)
#                            optimizer.step()
                            
                        if phase == 'train' :
                            for i,item in enumerate(losses):
                                if i < len(losses)-1:
                                    losses[item].backward(retain_graph = True)
                                else:
                                    losses[item].backward(retain_graph = False)
                            optimizer.step()
                            
                   
                    loss_vals = [losses[tag].item() for tag in losses]
                    total_loss = sum(loss_vals)
                    running_loss += total_loss
        
                    # verbose update
                    count += 1
                    if count % 1 == 0:
                        print("{:.4f} - Minibatch {} of {}".format(total_loss,count,total_num_minibatches))
                        #print("      cls:{:.4f} reg:{:.4f}  obj:{:.4f} rpn:{:.4f}"\
                        #      .format(loss_vals[0],loss_vals[1],loss_vals[2],loss_vals[3]))
                        if count % 500 == 0 and running_loss/(count*dataloaders[phase].batch_size) < best_loss:
                            print("New lowest loss, checkpoint saved.")
                            
                            # save checkpoint
                            PATH = "faster_rcnn_detrac_epoch_{}_{}.pt".format(epoch,count)
                            torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': running_loss
                            }, PATH)
    
                torch.cuda.empty_cache()
                
                
                # get epoch statistics
                running_loss = running_loss / dataset_sizes[phase]
                print("Epoch {} {} phase complete. Avg loss: {}".format(epoch,phase,running_loss))
                
                # save checkpoint
                PATH = "faster_rcnn_detrac_epoch_{}.pt".format(epoch)
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss
                }, PATH)
                
        return model

def load_model(checkpoint_file,model,optimizer):
    """
    Reloads a checkpoint, loading the model and optimizer state_dicts and 
    setting the start epoch
    """
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    return model,optimizer,epoch


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
    try:
        model
    except:
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained_backbone = True,num_classes = 14)
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
    try:
        trainloader
        testloader
    except:
        if False:
            label_dir = "C:\\Users\\derek\\Desktop\\UA Detrac\\DETRAC-Train-Annotations-XML-v3"
            image_dir = "C:\\Users\\derek\\Desktop\\UA Detrac\\Tracks"
        if True:
            label_dir = "/media/worklab/data_HDD/cv_data/UA_Detrac/DETRAC-Train-Annotations-XML-v3"
            image_dir = "/media/worklab/data_HDD/cv_data/UA_Detrac/DETRAC-train-data/Insight-MVT_Annotation_Train"
        
        print("Loading datasets")
        train_dataset = Track_Dataset(image_dir,label_dir,mode = "training")
        test_dataset = Track_Dataset(image_dir,label_dir,mode = "testing")
        
        # create training params
        params = {'batch_size': 4,
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
    
    
    #%%
    model = model.to(device)

    # all parameters are being optimized, not just fc layer
    #optimizer = optim.Adam(model.parameters(), lr=0.0001)
    optimizer = optim.SGD(model.parameters(), lr=0.001,momentum = 0.9)    
    # Decay LR by a factor of 0.5 every epoch
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    start_epoch = 0
    num_epochs = 10

#    checkpoint = "faster_rcnn_detrac_epoch_0.pt"
#    model,optimizer,start_epoch = load_model(checkpoint,model,optimizer)
        
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
    