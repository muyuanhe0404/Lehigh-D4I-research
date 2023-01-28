import cv2
import csv
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
import argparse
from pathlib import Path


cam = cv2.VideoCapture(r'../data/videos/trimed.mov')
#cam = cv2.VideoCapture(r'../data/videos/PC3_30%_3.avi')
cam_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT ))
cam_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH ))
projector_height = 500
projector_width = 1000
x_offset = 300
y_offset = 300
BinarizationThreshold = 5
ReferenceFrame = None
minContourArea = 250
g_counting = 0
b_counting = 0
frame_num = 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
input_size = 112
trans = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor()]) 

contour_area = 0.0
circularity = 0.0
diameter = 0
details = []





parser = argparse.ArgumentParser(description = '2d image ML code')
parser.add_argument('-i','--inference',  default = False, action='store_true',help = 'only process it on backend without showing extra information --> max speed////enter true or false' )
parser.add_argument('-s','--save',  default = False, action='store_true',help = 'save the info/cropped image into separate folders: naming convention "000_001.png"////enter true or false' )

args = parser.parse_args()
# print(args.save)
# print(args.inference)

# Utils for contour detections
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
    

## vvvvvvvvv********** ML prepartion ***********vvvvvvvvv ##

##  ******************Shen************* ##

def im_crop(frame_img, loc_x, loc_y,  width, height):
    box_size = max(width, height)
    roi = frame_img[max(int(loc_x - box_size /2 ), 0) : min(int(loc_x + box_size/2), cam_height), 
                    max(int(loc_y - box_size/2 ), 0) : min(int(loc_y + box_size/2 ), cam_width)]
    return roi

def load_ML(cnn_filename):

    # # Initialize network
    model_net = models.resnet50(pretrained=True)
    model_net = model_net.cuda() if torch.cuda.is_available() else model_net
    num_ftrs = model_net.fc.in_features
    model_net.fc = nn.Linear(num_ftrs, 3)
    model_net.fc = nn.Sequential(nn.Linear(num_ftrs, 512),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(512, 2))
    model_net.fc = model_net.fc.cuda() if torch.cuda.is_available() else model_net.fc
    trained_model_PATH = cnn_filename
    #  Load model details
    model_net.load_state_dict(torch.load(trained_model_PATH, map_location=torch.device('cpu')))
    model_net.eval()

    return model_net

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

def CNN_classify(cnn_classifier, single_cell_cropped, input_size = 112):
    with torch.no_grad():
        model_input = trans(single_cell_cropped)
        model_input = model_input.unsqueeze(0).to(device)
        output =  cnn_classifier(model_input)
        score_output = F.softmax(output)
        score, pred = torch.max(score_output, 1)
        label_pred = 'PC3' if pred.data == 0 else 'WBC' # ToDO: annotation library indicating the cell type

    return label_pred, score.data

##  ******************M&D************* ##
def bad_good_process(data):
    data_resized = resize(data, (64, 64))
    flat_data = data_resized.flatten()
    final_data = flat_data.reshape(1, -1)
    print("process complete")
    return final_data

def good_save(data,contour,frame):
    p = Path('../data/good/'+ str(frame))
    p.mkdir(exist_ok=True)
    good = Image.fromarray(np.uint8(data)).convert('RGB')
    f1 = (str(p) + "/" + ("{}_{}").format(str(frame).zfill(3),str(contour).zfill(3)) + '.png')
    filename = ''.join(f1)
    good.save(filename,"png")
    print("this is a good crop \n")
    print("pass it to classifier")
                    
def bad_save(data,contour,frame):
    p = Path('../data/bad/'+ str(frame))
    p.mkdir(exist_ok=True)
    bad = Image.fromarray(np.uint8(data)).convert('RGB')
    f1 = (str(p) + "/" + ("{}_{}").format(str(frame).zfill(3),str(contour).zfill(3)) + '.png')
    filename = ''.join(f1)
    bad.save(filename,"png")
    print("this is a bad crop \n")

def get_reduced_noise_img(ReferenceFrame,data,GrayFrame,BinarizationThreshold):
    ReferenceFrame = cv2.resize(ReferenceFrame, (np.size(data,1), np.size(data, 0)))
    FrameDelta = cv2.absdiff(ReferenceFrame, GrayFrame)
    FrameThresh = cv2.threshold(FrameDelta, BinarizationThreshold, 255, cv2.THRESH_BINARY)[1]  #fix add closing operation
    closed_img = cv2.morphologyEx(FrameThresh, cv2.MORPH_CLOSE, (9, 9))
    reduced_noise_img = cv2.medianBlur(closed_img,5)
    return reduced_noise_img

def process_Frame(ReferenceFrame,GrayFrame):
    ref_dist = cdf(ReferenceFrame)
    tgt_dist = cdf(GrayFrame)
    processed_tgt = hist_matching(tgt_dist, ref_dist, GrayFrame)
    GrayFrame = cv2.GaussianBlur(processed_tgt,(3,3),3)
    return GrayFrame

def get_croped_and_centriod(contours,data):
    g,y,w,h = cv2.boundingRect(contours)
    a, b = int((int(g)+int(w/2))), int((int(y) + int(h/2)))                
    loc_x, loc_y = b, a
    crop_img = im_crop(data, loc_x, loc_y, w, h)
    return crop_img, a, b

def visualization(times,data):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Elapsed Time:" + '{:5.2f}'.format(time.time() - times)
    cv2.putText(data, text, (10, 30), font, 0.8, (0, 0, 0), 1)
    #imD = cv2.resize(data, (960, 540))  
    #cv2.imshow('XiCAM example', imD)
    cv2.waitKey(1)

def process_classifier(image,classifier):
    single_cell_image = Image.fromarray(np.uint8(crop_img)).convert('RGB')
    # APPLY ML model and get prediction
    t = time.process_time()
    pred, score = CNN_classify(cnn_classifier, single_cell_image)
    elapsed_time = time.process_time() - t
    print(elapsed_time)
    return pred,score

def resize_img(data):
    gray_or_rgb = data.ndim;
    if gray_or_rgb == 3:
        GrayFrame = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    elif gray_or_rgb == 2:
        GrayFrame = data
    return GrayFrame

def visualize_video(frame):
    plt.figure()
    plt.imshow(frame)

def feature_save (details):
    all_data = []
    labels = ['Name', 'Type', 'Location', 'Area', 'Diameter', 'Circularity']
    all_data = [labels]
    all_data.extend(details)
    filename = "features.csv"
    with open("../data/features/"+filename, "w", encoding="utf-8") as f:
        f_csv = csv.writer(f)
        f_csv.writerows(all_data)
        f.close()
    
## ^^^^^^^^^^^^^^****** ML prepartion ******^^^^^^^^^^^^^^^^ ##
cnn_classifier = load_ML('../checkpoints/1011wbc-pc3.pt')
svm_scorer = pickle.load(open("../checkpoints/newone.sav", 'rb'))#load svm model


try:
    
    print('Starting video. Press CTRL+C to exit.')
    t0 = time.time()
    while True:
        check, img = cam.read()# get data and pass them from camera to img
        if not check:
            feature_save(details)
            break
        visualize_video(img)
        data = np.asarray(img)
        GrayFrame = resize_img(data)
        if ReferenceFrame is None:
            ReferenceFrame= GrayFrame
        img_bw = cc_prc98(GrayFrame)
        GrayFrame = process_Frame(ReferenceFrame,GrayFrame)
        key_reset = cv2.waitKey(1)
        if key_reset == ord('u'):
                ReferenceFrame = GrayFrame
                continue
        reduced_noise_img = get_reduced_noise_img(ReferenceFrame,data,GrayFrame,BinarizationThreshold)
        #show_full_frame(reduced_noise_img)
        contours, h = cv2.findContours(reduced_noise_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        x = 0
        i3 = np.zeros((cam_height, cam_width), np.uint8) # camera size
        debugging_info = data.copy() # camera size
        batch = []
        for _ in contours:
            x += 1
            if (x < len(contours)):
                if(cv2.contourArea(contours[x]) > minContourArea):
                    g,y,w,h = cv2.boundingRect(contours[x])
                    crop_img, a, b = get_croped_and_centriod(contours[x],img_bw)
                    svm_predict = svm_scorer.predict(bad_good_process(crop_img))
                    svm_proba = svm_scorer.predict_proba(bad_good_process(crop_img))
                    if svm_predict == 1:
                        if args.save:
                            #good_save(crop_img,x,frame_num)
                            contour_area = cv2.contourArea(contours[x])
                            circularity = (4 * np.pi * contour_area )/((cv2.arcLength(contours[x],True) * cv2.arcLength(contours[x],True)))
                            diameter = w
                            name = ("{}_{}").format(str(frame_num).zfill(3),str(x).zfill(3))
                            #Type = pred
                            Location = (( "{:3d}, {:3d}").format(a,b))
                            details.append([name,Location,contour_area,diameter,circularity])
                        batch.append(bad_good_process(crop_img))
                    
        if len(batch) != 0:
            pred,score = process_classifier(batch,cnn_classifier) 
        

        frame_num  = frame_num + 1         
        if not args.inference:
            #imS = cv2.resize(debugging_info, (960, 540))  
            #cv2.imshow('visualization',imS)       
            cv2.imshow('visualization',debugging_info)
        projection = np.zeros((cam_height,cam_width), np.uint8) # Creating a dark square with NUMPY camera size

        image = cv2.resize(i3, (projector_width, projector_height))   # Resize frame to projector size

        projection[x_offset:image.shape[0]+x_offset, y_offset:image.shape[1]+y_offset] = image  #Pasting the 'image' into the projector location
        #show_full_frame(projection)
        key = cv2.waitKey(2)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
        visualization(t0,data)
except KeyboardInterrupt:
    cv2.destroyAllWindows()
print('Done.')