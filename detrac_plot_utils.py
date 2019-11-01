import os
import time
import numpy as np

import torch
import cv2
import PIL
from PIL import Image
from torch.utils import data
import matplotlib.pyplot as plt


def pil_to_cv(pil_im):
    """ convert PIL image to cv2 image"""
    open_cv_image = np.array(pil_im) 
    # Convert RGB to BGR 
    return open_cv_image[:, :, ::-1] 


def plot_text(im,offset,cls,idnum,class_colors):
    """ Plots filled text box on original image, 
        utility function for plot_bboxes_2d """
    
    text = "{}: {}".format(idnum,cls)
    
    font_scale = 1.0
    font = cv2.FONT_HERSHEY_PLAIN
    
    # set the rectangle background to white
    rectangle_bgr = class_colors[cls]
    
    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    
    # set the text start position
    text_offset_x = int(offset[0])
    text_offset_y = int(offset[1])
    # make the coords of the box with a small padding of two pixels
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width - 2, text_offset_y - text_height - 2))
    cv2.rectangle(im, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(im, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)


def plot_bboxes_2d(im,label):
    """ Plots rectangular bboxes on image and returns image
    im - cv2 or PIL style image (function converts to cv2 style image)
    label - for one frame, in the form output by parse_label_file 
    bbox_im -  cv2 im with bboxes and labels plotted
    """
    
    # check type and convert PIL im to cv2 im if necessary
    assert type(im) in [np.ndarray, PIL.PngImagePlugin.PngImageFile], "Invalid image format"
    if type(im) == PIL.PngImagePlugin.PngImageFile:
        im = pil_to_cv(im)
    cv_im = im.copy() 
    
    class_colors = {
            'Cyclist': (255,150,0),
            'Pedestrian':(255,100,0),
            'Person':(255,50,0),
            'Car': (0,255,150),
            'Van': (0,255,100),
            'Truck': (0,255,50),
            'Tram': (0,100,255),
            'Misc': (0,50,255),
            'DontCare': (200,200,200)}
    
    for det in label:
        bbox = det['bbox2d']
        cls = det['class']
        idnum = det['id']
        
        cv2.rectangle(cv_im,(bbox[0],bbox[1]),(bbox[2],bbox[3]), class_colors[cls], 1)
        if cls != 'DontCare':
            plot_text(cv_im,(bbox[0],bbox[1]),cls,idnum,class_colors)
    return cv_im


def get_coords_3d(det_dict,P):
    """ returns the pixel-space coordinates of an object's 3d bounding box
        computed from the label and the camera parameters matrix
        for the idx object in the current frame
        det_dict - object representing one detection
        P - camera calibration matrix
        bbox3d - 8x2 numpy array with x,y coords for ________ """     
    # create matrix of bbox coords in physical space 

    l = det_dict['dim'][0]
    w = det_dict['dim'][1]
    h = det_dict['dim'][2]
    x_pos = det_dict['pos'][0]
    y_pos = det_dict['pos'][1]
    z_pos = det_dict['pos'][2]
    ry = det_dict['rot_y']
    cls = det_dict['class']
        
        
    # in absolute space (meters relative to obj center)
    obj_coord_array = np.array([[l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2],
                                [0,0,0,0,-h,-h,-h,-h],
                                [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]])
    
    # apply object-centered rotation here
    R = np.array([[cos(ry),0,sin(ry)],[0,1,0],[-sin(ry),0,cos(ry)]])
    rotated_corners = np.matmul(R,obj_coord_array)
    
    rotated_corners[0,:] += x_pos
    rotated_corners[1,:] += y_pos
    rotated_corners[2,:] += z_pos
    
    # transform with calibration matrix
    # add 4th row for matrix multiplication
    zeros = np.zeros([1,np.size(rotated_corners,1)])
    rotated_corners = np.concatenate((rotated_corners,zeros),0)

    
    pts_2d = np.matmul(P,rotated_corners)
    pts_2d[0,:] = pts_2d[0,:] / pts_2d[2,:]        
    pts_2d[1,:] = pts_2d[1,:] / pts_2d[2,:] 
    
    # apply camera space rotation here?
    return pts_2d[:2,:] ,pts_2d[2,:], rotated_corners

    
def draw_prism(im,coords,color):
    """ draws a rectangular prism on a copy of an image given the x,y coordinates 
    of the 8 corner points, does not make a copy of original image
    im - cv2 image
    coords - 2x8 numpy array with x,y coords for each corner
    prism_im - cv2 image with prism drawn"""
    prism_im = im.copy()
    coords = np.transpose(coords).astype(int)
    #fbr,fbl,rbl,rbr,ftr,ftl,frl,frr
    edge_array= np.array([[0,1,0,1,1,0,0,0],
                          [1,0,1,0,0,1,0,0],
                          [0,1,0,1,0,0,1,1],
                          [1,0,1,0,0,0,1,1],
                          [1,0,0,0,0,1,0,1],
                          [0,1,0,0,1,0,1,0],
                          [0,0,1,0,0,1,0,1],
                          [0,0,0,1,1,0,1,0]])

    # plot lines between indicated corner points
    for i in range(0,8):
        for j in range(0,8):
            if edge_array[i,j] == 1:
                cv2.line(prism_im,(coords[i,0],coords[i,1]),(coords[j,0],coords[j,1]),color,1)
    return prism_im


def plot_bboxes_3d(im,label,P, style = "normal"):
    """ Plots rectangular prism bboxes on image and returns image
    im - cv2 or PIL style image (function converts to cv2 style image)
    label - for one frame, in the form output by parse_label_file
    P - camera calibration matrix
    bbox_im -  cv2 im with bboxes and labels plotted
    style - string, "ground_truth" or "normal"  ground_truth plots boxes as white
    """
        
    # check type and convert PIL im to cv2 im if necessary
    assert type(im) in [np.ndarray, PIL.PngImagePlugin.PngImageFile], "Invalid image format"
    if type(im) == PIL.PngImagePlugin.PngImageFile:
        im = pil_to_cv(im)
    cv_im = im.copy() 
    
    class_colors = {
            'Cyclist': (255,150,0),
            'Pedestrian':(200,800,0),
            'Person':(160,30,0),
            'Car': (0,255,150),
            'Van': (0,255,100),
            'Truck': (0,255,50),
            'Tram': (0,100,255),
            'Misc': (0,50,255),
            'DontCare': (200,200,200)}
    
    for i in range (0,len(label)):
        if label[i]['pos'][2] > 2 and label[i]['truncation'] < 1:
            cls = label[i]['class']
            idnum = label[i]['id']
            if cls != "DontCare":
                bbox_3d,_,_ = get_coords_3d(label[i],P)
                if style == "ground_truth": # for plotting ground truth and predictions
                    cv_im = draw_prism(cv_im,bbox_3d,(255,255,255))
                else:
                    cv_im = draw_prism(cv_im,bbox_3d,class_colors[cls])
                    plot_text(cv_im,(bbox_3d[0,4],bbox_3d[1,4]),cls,idnum,class_colors)
    return cv_im
