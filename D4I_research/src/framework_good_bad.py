import cv2
import numpy as np
import time
import os
import scipy
from scipy import stats
from skimage.exposure import cumulative_distribution
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import *
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import time
from skimage.transform import resize
import numpy as np

def cdf(im):
    c,b = cumulative_distribution(im)
    c = np.insert(c,0,[0]*b[0])
    c = np.append(c,[1]*(255-b[-1]))
    return c
def hist_matching(c,c_t, im):
    pixel = np.arange(256)
    new_pixels = np.interp(c,c_t,pixel)
    im = (np.reshape(new_pixels[im.ravel()],im.shape)).astype(np.uint8)
    return im
def show_full_frame(frame):
    cv2.namedWindow('Full Screen', cv2.WINDOW_FREERATIO)
    cv2.setWindowProperty('Full Screen', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Full Screen', frame)


def cc_prc98(bgr):
    # Basic color correct & white balance
    prc50, prc98 = np.percentile(bgr, [50, 98], axis=(0, 1))
    sc98 = 225.0 / (prc98 + 1e-8)

    bgr = bgr * sc98.astype(np.float32)

    # N to match C++, need to round before casting to uint8.
    # OpenCV casts use saturate_cast() which rounds before the actual cast
    bgr = np.clip(bgr, 0, 255)
    out = np.zeros(bgr.shape, np.uint8)
    np.rint(bgr, out=out, casting='unsafe')  # slightly faster than doing a round() directly
    return out

cam = cv2.VideoCapture(r'/Users/hmy/desktop/framework/video/trimed.mov')
cam_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT ))#1536
cam_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH ))#2048
projector_height = 500
projector_width = 1000
x_offset = 300
y_offset = 300
BinarizationThreshold = 5
ReferenceFrame = None
minContourArea = 250
g_counting = 0
b_counting = 0


def im_crop(frame_img, loc_x, loc_y,  width, height):
    box_size = max(width, height)
    roi = frame_img[max(int(loc_x - box_size /2 ), 0) : min(int(loc_x + box_size/2), cam_height), 
                    max(int(loc_y - box_size/2 ), 0) : min(int(loc_y + box_size/2 ), cam_width)]
    print(max(int(loc_x - box_size /2 ), 0))
    print(min(int(loc_x + box_size/2), cam_height))
    print(max(int(loc_y - box_size/2 ), 0))
    print(min(int(loc_y + box_size/2 ), cam_width))
    return roi


svm_scorer = pickle.load(open("/Users/hmy/desktop/framework/sav/newone.sav", 'rb'))#load svm model

try:
    print('Starting video. Press CTRL+C to exit.')
    t0 = time.time()
    while True:
        # get data and pass them from camera to img
        check, img = cam.read()
        if not check:
            break
        data = np.asarray(img)
        # resize image
        gray_or_rgb = data.ndim
        if gray_or_rgb ==3:
                GrayFrame = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        elif gray_or_rgb == 2:
            GrayFrame = data
        if ReferenceFrame is None:
            ReferenceFrame = GrayFrame
        ref_dist = cdf(ReferenceFrame)
        tgt_dist = cdf(GrayFrame)
        img_bw = cc_prc98(GrayFrame)
        processed_tgt = hist_matching(tgt_dist, ref_dist, GrayFrame)
        GrayFrame = cv2.GaussianBlur(processed_tgt,(3,3),3)
        key_reset = cv2.waitKey(1)
        if key_reset == ord('u'):
                ReferenceFrame = GrayFrame
                continue
        ReferenceFrame = cv2.resize(ReferenceFrame, (np.size(data,1), np.size(data, 0)))
        FrameDelta = cv2.absdiff(ReferenceFrame, GrayFrame)
        FrameThresh = cv2.threshold(FrameDelta, BinarizationThreshold, 255, cv2.THRESH_BINARY)[1]  #fix add closing operation
        closed_img = cv2.morphologyEx(FrameThresh, cv2.MORPH_CLOSE, (9, 9))
        reduced_noise_img = cv2.medianBlur(closed_img,5)
        #cv2.imshow('img_bw', reduced_noise_img)
       # filled_img = mo.area_closing(reduced_noise_img, 1000)
        contours, he = cv2.findContours(reduced_noise_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        #show_full_frame(reduced_noise_img)
        x=0
        i3 = np.zeros((cam_height, cam_width), np.uint8) # camera size

        #print(contours[0])
        #i3 = cv2.drawContours(img_bw,contours[0])
        #cv2.imshow("i3",i3)

        for i in contours:
            x+=1
            if (x<len(contours)):
                if(cv2.contourArea(contours[x])>minContourArea):
                    g,y,w,h = cv2.boundingRect(contours[x])
                    (a,b) = int((int(g)+int(w/2))), int((int(y) + int(h/2)))
                    print(data.shape)
                    loc_x, loc_y = b,a
                    crop_img = im_crop(img_bw, loc_x, loc_y, w, h)
                    cv2.imshow('single cell', crop_img)
                    data_resized = resize(crop_img, (64, 64))
                    flat_data = data_resized.flatten()
                    final_data = flat_data.reshape(1, -1)
                    print("process complete")
                    #print(single_cell_crop[0])
                    data = cv2.circle(data, (a, b), (50), (255, 255, 255), -1)
                    svm_input = final_data
                    print(type(svm_input))
                    svm_predict = svm_scorer.predict(svm_input)
                    print(svm_predict)
                    svm_proba = svm_scorer.predict_proba(svm_input)
                    print("svm_Predicted=",svm_proba)
                    if svm_predict == 1:
                        good = Image.fromarray(np.uint8(crop_img)).convert('RGB')
                        gc = str(g_counting)
                        f1 = ('/Users/hmy/desktop/framework/good/good'+gc +'.png')
                        filename = ''.join(f1)
                        print(f1)
                        print(filename)
                        good.save(filename,"png")
                        g_counting = g_counting + 1
                    # The actual projection
                        print("svm_Predicted = {}, this is a good crop",svm_proba)
                        i3 = cv2.circle(i3, (a, b), (50), (255), -1) 
                    else:
                        bad = Image.fromarray(np.uint8(crop_img)).convert('RGB')
                        bc = str(b_counting)
                        f2 = ('/Users/hmy/desktop/framework/bad/bad' + bc + '.png')
                        filename = ''.join(f2)
                        bad.save(filename,"png")
                        b_counting = b_counting + 1




        # Creating a dark square with NUMPY camera size
        f = np.zeros((cam_height,cam_width), np.uint8)
        # Resize frame to projector size
        image = cv2.resize(i3, (projector_width, projector_height)) # projector size
        # Pasting the 'image' into the projector location
        f[x_offset:image.shape[0]+x_offset, y_offset:image.shape[1]+y_offset] = image  # offsets need to be fixed
        # cv2.imshow('Image',image)
        #show_full_frame(f)
        key = cv2.waitKey(2)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Elapsed Time:" + '{:5.2f}'.format(time.time() - t0)
        cv2.putText(data, text, (10, 30), font, 0.8, (0, 0, 0), 1)
        #cv2.imshow('XiCAM example', data)
        #cv2.imshow('XiCAM example', data)
        cv2.waitKey(1)
except KeyboardInterrupt:
    cv2.destroyAllWindows()
print('Done.')