"""
This file provides a dataset class for working with the UA-detrac tracking dataset.
Provides:
    - plotting of 2D bounding boxes
    - transforms for training
    - training/testing loader mode (random images from across all tracks)
    - track mode - returns a single image, in order
    
It is assumed that image dir is a directory containing a subdirectory for each track
Label dir is a directory containing a bunch of label files
"""
label_dir = "C:\\Users\\derek\\Desktop\\UA Detrac\\DETRAC-Train-Annotations-XML-v3"
image_dir = "C:\\Users\\derek\\Desktop\\UA Detrac\\Tracks"

import os
import time
import numpy as np

import torch
import cv2
import PIL
from PIL import Image
from torch.utils import data
import matplotlib.pyplot as plt

from detrac_plot_utils import pil_to_cv

class Track_Dataset(data.Dataset):
    """
    Creates an object for referencing the UA-Detrac 2D object tracking dataset
    """
    
    def __init__(self, image_dir, label_dir):
        """ initializes object. By default, the first track (cur_track = 0) is loaded 
        such that next(object) will pull next frame from the first track"""

        # stores files for each set of images and each label
        dir_list = next(os.walk(image_dir))[1]
        track_list = [os.path.join(image_dir,item) for item in dir_list]
        label_list = [os.path.join(label_dir,item) for item in os.listdir(label_dir)]
        track_list.sort()
        label_list.sort()
        
        self.track_offsets = [0]
        self.all_data = []
        
        # parse and store all labels and image names in a list such that
        # all_data[i] returns dict with image name, label and other stats
        # track_offsets[i] retuns index of first frame of track[i[]]
        for i in range(0,len(track_list)):

            images = [os.path.join(track_list[i],frame) for frame in os.listdir(track_list[i])]
            images.sort() 
            labels = self.parse_labels(label_list[i])
            
            for j in range(len(images)):
                out_dict = {
                        'image':images[j],
                        'label':labels[j],
                        'track_len': len(images),
                        'track_num': i,
                        'frame_num_of_track':j
                        }
                self.all_data.append(out_dict)
            
            # index of first frame
            if i > 0:
                self.track_offsets.append(len(images)+self.track_offsets[i-1])
            
        # for keeping track of things
        self.cur_track =  None # int
        self.cur_frame = None
        self.num_tracks = len(track_list)
        self.total_num_frames = len(self.all_data)
        
        # in case it is later important which files are which
        self.track_list = track_list
        self.label_list = label_list
        
        # load track 0
        self.load_track(0)
        
        
    def load_track(self,idx):
        """moves to track indexed"""
        try:
            if idx >= self.num_tracks or idx < 0:
                raise Exception
                
            self.cur_track = idx
            # so that calling next will load frame 0 of that track
            self.cur_frame = self.track_offsets[idx]-1 
        except:
            print("Invalid track number")
            
    def num_tracks(self):
        """ return number of tracks"""
        return self.num_tracks
    
    def __next__(self):
        """get next frame and label from current track"""
        
                
        self.cur_frame = self.cur_frame + 1
        cur = self.all_data[self.cur_frame]
        im = Image.open(cur['image'])
        label = cur['label']
        track_len = cur['track_len']
        frame_num_of_track = cur['frame_num_of_track']
        track_num = cur['track_num']
        
        return im, label, frame_num_of_track, track_len, track_num


    def __len__(self):
        """ returns total number frames in all tracks"""
        return self.total_num_frames
    
    def __getitem__(self,index):
        """ returns item indexed from all frames in all tracks"""
        cur = self.all_data[index]
        im = Image.open(cur['image'])
        label = cur['label']
        frame_num_of_track = cur['frame_num_of_track']
        track_num = cur['track_num']
        
        return im, label
    
    def parse_labels(self,label):
        return [0 for i in range(0,10000)]
    
    def plot(self,track_idx,show_labels = False):
        """ plots all frames in track_idx as video"""

        self.load_track(track_idx)
        im,label,frame_num,track_len,_ = next(self)
        
        while True:
            
            print("Frame num {}, Track len {}".format(frame_num, track_len))
            cv_im = pil_to_cv(im)
            
            cv2.imshow("Frame",cv_im)
            key = cv2.waitKey(1) & 0xff
            #time.sleep(1/30.0)
            
            if key == ord('q'):
                break
            
            # load next frame
            im,label,frame_num,track_len,_ = next(self)
            if frame_num == 0:
                break
    
        cv2.destroyAllWindows()