"""
This file provides a dataset class for working with the UA-detrac tracking dataset.
Provides:
    - plotting of 2D bounding boxes
    - training/testing loader mode (random images from across all tracks) using __getitem__()
    - track mode - returns a single image, in order, using __next__()
    
It is assumed that image dir is a directory containing a subdirectory for each track sequence
Label dir is a directory containing a bunch of label files
"""

import os
import numpy as np

import cv2
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms

import xml.etree.ElementTree as ET


from detrac_plot_utils import pil_to_cv, plot_bboxes_2d

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
        
        self.im_tf = transforms.ToTensor()
        self.track_offsets = [0]
        self.track_metadata = []
        self.all_data = []
        
        # parse and store all labels and image names in a list such that
        # all_data[i] returns dict with image name, label and other stats
        # track_offsets[i] retuns index of first frame of track[i[]]
        for i in range(0,len(track_list)):

            images = [os.path.join(track_list[i],frame) for frame in os.listdir(track_list[i])]
            images.sort() 
            labels,metadata = self.parse_labels(label_list[i])
            self.track_metadata.append(metadata)
            
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
            if i < len(track_list) - 1:
                self.track_offsets.append(len(images)+self.track_offsets[i])
            
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
        """moves to track indexed by idx (int)."""
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
        """get next frame label, and a bit of other info from current track"""
                
        self.cur_frame = self.cur_frame + 1
        cur = self.all_data[self.cur_frame]
        im = Image.open(cur['image'])
        label = cur['label']
        track_len = cur['track_len']
        frame_num_of_track = cur['frame_num_of_track']
        track_num = cur['track_num']
        metadata = self.track_metadata[track_num]
        
        return im, label, frame_num_of_track, track_len, track_num, metadata


    def __len__(self):
        """ returns total number of frames in all tracks"""
        return self.total_num_frames
    
    def __getitem__(self,index):
        """ returns item indexed from all frames in all tracks"""
        cur = self.all_data[index]
        im = Image.open(cur['image'])
        label = cur['label']
        
        # convert image and label to tensors
        im = self.im_tf(im)
        all_bboxes = []
        all_cls = []
        for item in label:
            all_bboxes.append( torch.from_numpy(item['bbox']).float() )
            val = item['class_num']
            all_cls.append(  torch.tensor(val).long() )
        
        all_bboxes = torch.stack(all_bboxes,dim = 0)
        all_cls = torch.stack(all_cls)
        
        label = {
                'boxes': all_bboxes,
                'labels':all_cls
                }
        
        
        return im, label
    
    def parse_labels(self,label_file):
        """
        Returns a set of metadata (1 per track) and a list of labels (1 item per
        frame, where an item is a list of dictionaries (one dictionary per object
        with fields id, class, truncation, orientation, and bbox
        """
        
        class_dict = {
        'Sedan':0,
        'Hatchback':1,
        'Suv':2,
        'Van':3,
        'Police':4,
        'Taxi':5,
        'Bus':6,
        'Truck-Box-Large':7
        }
        
        
        tree = ET.parse(label_file)
        root = tree.getroot()
        
        # get sequence attributes
        seq_name = root.attrib['name']
        
        # get list of all frame elements
        frames = root.getchildren()
        
        # first child is sequence attributes
        seq_attrs = frames[0].attrib
        
        # second child is ignored regions
        ignored_regions = []
        for region in frames[1]:
            coords = region.attrib
            box = np.array([float(coords['left']),
                            float(coords['top']),
                            float(coords['left']) + float(coords['width']),
                            float(coords['top'])  + float(coords['height'])])
            ignored_regions.append(box)
        frames = frames[2:]
        
        # rest are bboxes
        all_boxes = []
        for frame in frames:
            frame_boxes = []
            boxids = frame.getchildren()[0].getchildren()
            for boxid in boxids:
                data = boxid.getchildren()
                coords = data[0].attrib
                stats = data[1].attrib
                bbox = np.array([float(coords['left']),
                                float(coords['top']),
                                float(coords['left']) + float(coords['width']),
                                float(coords['top'])  + float(coords['height'])])
                det_dict = {
                        'id':int(boxid.attrib['id']),
                        'class':stats['vehicle_type'],
                        'class_num':class_dict[stats['vehicle_type']],
                        'color':stats['color'],
                        'orientation':float(stats['orientation']),
                        'truncation':float(stats['truncation_ratio']),
                        'bbox':bbox
                        }
                
                frame_boxes.append(det_dict)
            all_boxes.append(frame_boxes)
        
        sequence_metadata = {
                'sequence':seq_name,
                'seq_attributes':seq_attrs,
                'ignored_regions':ignored_regions
                }
        return all_boxes, sequence_metadata
    
    def plot(self,track_idx,SHOW_LABELS = True):
        """ plots all frames in track_idx as video
            SHOW_LABELS - if True, labels are plotted on sequence
            track_idx - int    
        """

        self.load_track(track_idx)
        im,label,frame_num,track_len,track_num,metadata = next(self)
        
        while True:
            cv_im = pil_to_cv(im)
            
            if SHOW_LABELS:
                cv_im = plot_bboxes_2d(cv_im,label,metadata['ignored_regions'])
                
            cv2.imshow("Frame",cv_im)
            key = cv2.waitKey(1) & 0xff
            #time.sleep(1/30.0)
            
            if key == ord('q'):
                break
            
            # load next frame
            im,label,frame_num,track_len,track_num,metadata = next(self)
            if frame_num == track_len - 1:
                break
    
        cv2.destroyAllWindows()
        


#### Test script here
label_dir = "C:\\Users\\derek\\Desktop\\UA Detrac\\DETRAC-Train-Annotations-XML-v3"
image_dir = "C:\\Users\\derek\\Desktop\\UA Detrac\\Tracks"
test = Track_Dataset(image_dir,label_dir)
test.plot(0)
temp = test[0]

