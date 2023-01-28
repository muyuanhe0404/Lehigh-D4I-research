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


print(torch.__version__)
# cam = cv2.VideoCapture(r'./data/videos/trimed.mov')
# cam = cv2.VideoCapture(r'../data/videos/PC3_flow_3.avi')
cam = cv2.VideoCapture(r'../data/videos/1_convert.avi')
# cam = cv2.VideoCapture(r'../data/videos/1.avi')
cam_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
cam_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
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
hct_counting = 0
t_counting = 0
frame_num = 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
input_size = 64
trans = transforms.Compose(
    [transforms.Resize((input_size, input_size)), transforms.ToTensor()])

contour_area = 0.0
circularity = 0.0
diameter = 0
details = []


parser = argparse.ArgumentParser(description='2d image ML code')
parser.add_argument('-i', '--inference',  default=False, action='store_true',
                    help='only process it on backend without showing extra information --> max speed////enter true or false')
parser.add_argument('-s', '--save',  default=False, action='store_true',
                    help='save the info/cropped image into separate folders: naming convention "000_001.png"////enter true or false')

args = parser.parse_args()
# print(args.save)
# print(args.inference)

# Utils for contour detections


def cdf(im):
    c, b = cumulative_distribution(im)
    c = np.insert(c, 0, [0]*b[0])
    c = np.append(c, [1]*(255-b[-1]))
    return c


def hist_matching(c, c_t, im):
    pixel = np.arange(256)
    new_pixels = np.interp(c, c_t, pixel)
    im = (np.reshape(new_pixels[im.ravel()], im.shape)).astype(np.uint8)
    return im


def show_full_frame(frame):
    cv2.namedWindow('Full Screen', cv2.WINDOW_FREERATIO)
    cv2.setWindowProperty(
        'Full Screen', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Full Screen', frame)


## vvvvvvvvv********** ML prepartion ***********vvvvvvvvv ##

##  ******************Shen************* ##

def im_crop(frame_img, loc_x, loc_y,  width, height):
    box_size = max(width, height)
    # box_size = input_size
    roi = frame_img[max(int(loc_x - box_size / 2), 0): min(int(loc_x + box_size/2), cam_height),
                    max(int(loc_y - box_size/2), 0): min(int(loc_y + box_size/2), cam_width)]
    return roi


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


def cc_prc98(bgr):
    # Basic color correct & white balance
    prc50, prc98 = np.percentile(bgr, [50, 98], axis=(0, 1))
    sc98 = 225.0 / (prc98 + 1e-8)

    bgr = bgr * sc98.astype(np.float32)

    # N to match C++, need to round before casting to uint8.
    # OpenCV casts use saturate_cast() which rounds before the actual cast
    bgr = np.clip(bgr, 0, 255)
    out = np.zeros(bgr.shape, np.uint8)
    # slightly faster than doing a round() directly
    np.rint(bgr, out=out, casting='unsafe')
    return out

# def CNN_classify(cnn_classifier, single_cell_cropped, input_size = 112):


def CNN_classify(cnn_classifier, batch_cell_cropped, input_size=112):
    with torch.no_grad():
        # model_input = trans(batch_cell_cropped)
        # model_input = model_input.unsqueeze(0).to(device)
        output = cnn_classifier(batch_cell_cropped)
        print(output.shape)
        score_output = F.softmax(output)
        score, pred = torch.max(score_output, 1)
        # label_pred = 'PC3' if pred.data == 0 else 'WBC' # ToDO: annotation library indicating the cell type

    return pred, score.data

##  ******************M&D************* ##


def bad_good_process(data):
    data_resized = resize(data, (64, 64))
    flat_data = data_resized.flatten()
    final_data = flat_data.reshape(1, -1)
    print("process complete")
    return final_data


def good_save(data, frame):
    p = Path('../data/good/' + str(frame))
    p.mkdir(exist_ok=True)
    good = Image.fromarray(np.uint8(data)).convert('RGB')
    f1 = (str(p) + "/" + ("{}_{}").format(str(frame).zfill(3),
          str(frame).zfill(3)) + '.png')
    filename = ''.join(f1)
    good.save(filename, "png")
    print("this is a good crop \n")
    print("pass it to classifier")


def bad_save(data, contour, frame):
    p = Path('../data/bad/' + str(frame))
    p.mkdir(exist_ok=True)
    bad = Image.fromarray(np.uint8(data)).convert('RGB')
    f1 = (str(p) + "/" + ("{}_{}").format(str(frame).zfill(3),
          str(contour).zfill(3)) + '.png')
    filename = ''.join(f1)
    bad.save(filename, "png")
    print("this is a bad crop \n")


def get_reduced_noise_img(ReferenceFrame, data, GrayFrame, BinarizationThreshold):
    ReferenceFrame = cv2.resize(
        ReferenceFrame, (np.size(data, 1), np.size(data, 0)))
    FrameDelta = cv2.absdiff(ReferenceFrame, GrayFrame)
    FrameThresh = cv2.threshold(FrameDelta, BinarizationThreshold, 255, cv2.THRESH_BINARY)[
                                1]  # fix add closing operation
    closed_img = cv2.morphologyEx(FrameThresh, cv2.MORPH_CLOSE, (9, 9))
    reduced_noise_img = cv2.medianBlur(closed_img, 5)
    return reduced_noise_img


def process_Frame(ReferenceFrame, GrayFrame):
    ref_dist = cdf(ReferenceFrame)
    tgt_dist = cdf(GrayFrame)
    processed_tgt = hist_matching(tgt_dist, ref_dist, GrayFrame)
    GrayFrame = cv2.GaussianBlur(processed_tgt, (3, 3), 3)
    return GrayFrame


def get_croped_and_centriod(contours, data):
    g, y, w, h = cv2.boundingRect(contours)
    a, b = int((int(g)+int(w/2))), int((int(y) + int(h/2)))
    loc_x, loc_y = b, a
    crop_img = im_crop(data, loc_x, loc_y, w, h)
    return crop_img, a, b


def visualization(times, data):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Elapsed Time:" + '{:5.2f}'.format(time.time() - times)
    cv2.putText(data, text, (10, 30), font, 0.8, (0, 0, 0), 1)
    # imD = cv2.resize(data, (960, 540))
    # cv2.imshow('XiCAM example', imD)
    cv2.waitKey(1)


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


def resize_img(data):
    gray_or_rgb = data.ndim;
    if gray_or_rgb == 3:
        GrayFrame = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    elif gray_or_rgb == 2:
        GrayFrame = data
    return GrayFrame




## ^^^^^^^^^^^^^^****** ML prepartion ******^^^^^^^^^^^^^^^^ ##
# cnn_classifier = load_ML('./checkpoints/1011wbc-pc3.pt')
# cnn_classifier = load_ML('./checkpoints/1228wbc-pc3-neg.pt') //original
cnn_classifier = load_ML('../checkpoints/0323hct-pc3-jur.pt')
# svm_scorer = pickle.load(open("./checkpoints/oct26.sav", 'rb'))#load svm model
dirname = "../data/0321PEG_Fixed/PC3"
batch = torch.empty(1,3,64,64)
final = []
check = 0
cropped_img = []
for fname in sorted(os.listdir(dirname)):
    check += 1
    print(check)
    im = Image.open(os.path.join(dirname, fname))
    crop_img = np.asarray(im)
    cropped_img.append(crop_img)
    batch_cell_image = Image.fromarray(np.uint8(crop_img)).convert('RGB')
    mo_input = trans(batch_cell_image)
    mo_input = mo_input.unsqueeze(0).to(device)
    batch = torch.cat((batch, mo_input))
if(batch.shape[0] > 1):
    batch = batch[1:, :, :, :]
pred, score = process_classifier(batch, cnn_classifier)
cell_predicted = 'Nothing'
if(len(cropped_img)):
    for index, pred in np.ndenumerate(pred.numpy()):
        t_counting += 1
        print(index)
        if pred == 2:
            if(len(cropped_img)):
                good_save(cropped_img[int(str(index[0]))],t_counting)
                print("pc3")
                hct_counting += 1

        print("pc3:", hct_counting)
        print("Total:", t_counting)



    

