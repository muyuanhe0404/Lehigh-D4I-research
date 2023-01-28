import cv2
import csv
import numpy as np
import time
import os
import scipy
from scipy import stats
from skimage.exposure import cumulative_distribution
import pandas
import timeit
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
import csv
import subprocess
import sys

start = timeit.default_timer()
cam = cv2.VideoCapture(r'../data/videos/1.mov')
cam_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT ))
cam_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH ))
print(cam_height)
print(cam_width)
projector_height = 500
projector_width = 1000
x_offset = 300
y_offset = 300
BinarizationThreshold = 5
ReferenceFrame = None
minContourArea = 1000
g_counting = 0
pc3_counting = 0
b_counting = 0
wbc_counting = 0
t_counting = 0
frame_num = 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
input_size = 64
trans = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor()]) 

contour_area = 0.0
circularity = 0.0
diameter = 0
details = []
a = []
b = []
contour_n = []
fram_m = []
type = []
w = []
h = []

# Utils for contour detections
def load_ML(cnn_filename):
    
    # # Initialize network
    model_net = models.resnet50(pretrained=True)
    # model_net = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

    # model_net.classifier[1] = nn.Linear(model_net.last_channel, 3)
    # model_net = model_net.cuda() if torch.cuda.is_available() else model_net
    num_ftrs = model_net.fc.in_features
    model_net.fc = nn.Linear(num_ftrs, 2)
    model_net.fc = nn.Sequential(nn.Linear(num_ftrs, 512),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(512, 3))
    model_net.fc = model_net.fc.cuda() if torch.cuda.is_available() else model_net.fc
    trained_model_PATH = cnn_filename
    #  Load model details
    model_net.load_state_dict(torch.load(
        trained_model_PATH, map_location=torch.device('cpu')))
    model_net.eval()

    return model_net
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
    
def im_crop(frame_img, loc_x, loc_y,  width, height):
    box_size = max(width, height)
    # box_size = input_size
    roi = frame_img[max(int(loc_x - box_size /2 ), 0) : min(int(loc_x + box_size/2), cam_height), 
                    max(int(loc_y - box_size/2 ), 0) : min(int(loc_y + box_size/2 ), cam_width)]
    return roi
def process_classifier(image, classifier):
    # batch_cell_image = Image.fromarray(np.uint8(batch)).convert('RGB')
    # single_cell_image = Image.fromarray(np.uint8(image)).convert('RGB')
    # APPLY ML model and get prediction
    t = time.process_time()
    pred, score = CNN_classify(cnn_classifier, image)
    # pred, score = CNN_classify(cnn_classifier, single_cell_image)
    elapsed_time = time.process_time() - t
    # print(elapsed_time)
    return pred, score
def CNN_classify(cnn_classifier, batch_cell_cropped, input_size=112):
    with torch.no_grad():
        # model_input = trans(batch_cell_cropped)
        # model_input = model_input.unsqueeze(0).to(device)
        output = cnn_classifier(batch_cell_cropped)
        score_output = F.softmax(output)
        score, pred = torch.max(score_output, 1)
        # label_pred = 'PC3' if pred.data == 0 else 'WBC' # ToDO: annotation library indicating the cell type

    return pred, score.data
def good_save(data,count,frame):
    p = Path('../data/good_ori/'+ str(frame))
    p.mkdir(exist_ok=True)
    good = Image.fromarray(np.uint8(data)).convert('RGB')
    f1 = (str(p) + "/" + ("{}_{}").format(str(frame).zfill(3),str(count).zfill(3)) + '.png')
    filename = ''.join(f1)
    good.save(filename,"png")
    print("this is a good crop \n")
                    
def bad_save(data,frame):
    p = Path('../data/bad_ori/'+ str(frame))
    p.mkdir(exist_ok=True)
    bad = Image.fromarray(np.uint8(data)).convert('RGB')
    f1 = (str(p) + "/" + ("{}_{}").format(str(frame).zfill(3),str(frame).zfill(3)) + '.png')
    filename = ''.join(f1)
    bad.save(filename,"png")
    print("this is a bad crop \n")

def good_learning_save(data,frame):
    p = Path('../data/good_learning/'+ str(frame))
    p.mkdir(exist_ok=True)
    good = Image.fromarray(np.uint8(data)).convert('RGB')
    f1 = (str(p) + "/" + ("{}").format(str(frame).zfill(3)) + '.png')
    filename = ''.join(f1)
    good.save(filename,"png")
    print("save for learning \n")

                    
def bad_learning_save(data,frame):
    p = Path('../data/bad_learning/'+ str(frame))
    p.mkdir(exist_ok=True)
    bad = Image.fromarray(np.uint8(data)).convert('RGB')
    f1 = (str(p) + "/" + ("{}").format(str(frame).zfill(3)) + '.png')
    filename = ''.join(f1)
    bad.save(filename,"png")
    print("save for learning \n")

def load_csv(files):
    file = open(files)
    csvreader = csv.reader(file)
    header = next(csvreader)
    a = []
    b = []
    fram_n = []
    type = []
    w = []
    h = []
    for row in csvreader:
        a.append(int(float(row[0])))
        b.append(int(float(row[1])))
        fram_n.append(int(row[2]))
        type.append(str(row[3]))
        w.append(int(float(row[4])))
        h.append(int(float(row[5])))
    # print(a)   
    # print(b)
    # print(contour_n)
    # print(type)
    # print(w)
    # print(h)
    file.close()
    return a,b,fram_n,type,w,h


FPS = 30 #  frames per second
FLOW_TIME = 2 # second
STOP_PERIOD = 10 # seconds
PROJECTION_TIME = 21 # seconds

FLOW_FRAME = FLOW_TIME * FPS
PAUSE_PACE = (FLOW_TIME + STOP_PERIOD + PROJECTION_TIME) * FPS # FRAMES
INITIAL_PHASE =  int(1 + FLOW_TIME + STOP_PERIOD/2)* FPS


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
#VISUALIZE_SIZE = (width, height)
VISUALIZE_SIZE = (640, 480)
fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
out = cv2.VideoWriter('1.avi', fourcc, 20.0, VISUALIZE_SIZE)

dl_command = "python model_tester.py"
subprocess.run(dl_command, shell=True)

cnn_classifier = load_ML('../checkpoints/0323hct-pc3-jur.pt')
files = "../data/features/dl_coordinate.csv"
dirname = "../data/videos/check"

a,b,fram_n,type,w,h = load_csv(files)
count_g = 0
count_b = 0
final = []
check = 0
cropped_img = []
batch = torch.empty(1,3,64,64)
pc3_counting = 0
for fname in sorted(os.listdir(dirname)):
    if not fname.startswith('.') and os.path.isfile(os.path.join(dirname, fname)):
        check += 240
        print(check)
        im = Image.open(os.path.join(dirname, fname))
        data = np.asarray(im)
        count = 0
        for idx, value in enumerate(fram_n):
            if check == value:
                good_learning_save(data,fram_n[idx])
                crop_img = im_crop(data, (b[idx]+(h[idx]/2)), (a[idx]+(w[idx]/2)), w[idx], h[idx])     
                good_save(crop_img,count,fram_n[idx])
                cropped_img.append(crop_img)
                batch_cell_image = Image.fromarray(np.uint8(crop_img)).convert('RGB')
                mo_input = trans(batch_cell_image)
                mo_input = mo_input.unsqueeze(0).to(device)
                batch = torch.cat((batch, mo_input))
                count += 1
    if(batch.shape[0] > 1):
        batch = batch[1:, :, :, :]
    pred, score = process_classifier(batch, cnn_classifier)
    cell_predicted = 'Nothing'
    if(len(cropped_img)):
        for index, pred in np.ndenumerate(pred.numpy()):
            t_counting += 1
            if pred == 2:
                if(len(cropped_img)):
                    print("pc3")
                    pc3_counting += 1
                    print("pc3:", pc3_counting)
                    print("Total:", t_counting)
            else:
                print("not pc3")
                print("Total:", t_counting)
stop = timeit.default_timer()
print('all Time: ', stop - start)  
# #runfile('framwork_combined_2_pause2.py', args='-i')