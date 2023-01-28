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


#cam = cv2.VideoCapture(r'./data/videos/trimed.mov')
cam = cv2.VideoCapture(r'./data/videos/PC3_flow_3.avi')
cam_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT ))
cam_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH ))
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
    p = Path('./data/good/'+ str(frame))
    p.mkdir(exist_ok=True)
    good = Image.fromarray(np.uint8(data)).convert('RGB')
    f1 = (str(p) + "/" + ("{}_{}").format(str(frame).zfill(3),str(contour).zfill(3)) + '.png')
    filename = ''.join(f1)
    good.save(filename,"png")
    print("this is a good crop \n")
    print("pass it to classifier")
                    
def bad_save(data,contour,frame):
    p = Path('./data/bad/'+ str(frame))
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
    # imD = cv2.resize(data, (960, 540))  
    # cv2.imshow('XiCAM example', imD)
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
def parallel (cell,minContourArea,img_bw,cnn_classifier,frame_num,x,i3,g_counting,pc3_counting,debugging_info,details):
    if(cv2.contourArea(cell) > minContourArea):
        g,y,w,h = cv2.boundingRect(cell)
        crop_img, a, b = get_croped_and_centriod(cell,img_bw)
        #cv2.imshow('cropped cell', crop_img)
        svm_predict = svm_scorer.predict(bad_good_process(crop_img))
        svm_proba = svm_scorer.predict_proba(bad_good_process(crop_img))
        #print("svm_Predicted=",svm_proba)
        pred,score = process_classifier(crop_img,cnn_classifier) 
        if svm_predict == 1:
            if args.save:
                #good_save(crop_img,x,frame_num)
                contour_area = cv2.contourArea(cell)
                circularity = (4 * np.pi * contour_area )/((cv2.arcLength(cell,True) * cv2.arcLength(cell,True)))
                diameter = w
                name = ("{}_{}").format(str(frame_num).zfill(3),str(x).zfill(3))
                x += 1
                Type = pred
                Location = (( "{:3d}, {:3d}").format(a,b))
                details.append([name,Type,Location,contour_area,diameter,circularity])
                            
                # good_save(crop_img,g_counting)
                g_counting = g_counting + 1
                i3 = cv2.circle(i3, (a, b), (50), (255), -1)
            if pred == "PC3": # A propoer condition, here e.g. "PC3"
                                #print("Get prediction with classification score{}, start projection".format(score)) # The actual projection
                i3 = cv2.circle(i3, (a, b), (50), (255), -1) 
                pc3_counting += 1
                if not args.inference:
                    debugging_info = cv2.rectangle(debugging_info, (g,y), ((g+w),(y+h)), (0,255,0), 3) #green means good PC3
            else:
                print("not PC3")

                if not args.inference:
                    debugging_info = cv2.rectangle(debugging_info, (g,y), ((g+w),(y+h)), (0,0,255), 3) #red means good something else
        else:
            # if args.save:
            #bad_save(crop_img,x,frame_num)
            if not args.inference:
                debugging_info = cv2.rectangle(debugging_info, (g,y), ((g+w),(y+h)), (255,0,0), 3) #blue means bad anything
        if not args.inference:
            debugging_info = cv2.putText(debugging_info, (pred + ":" + ( "{:.2f}").format(score.item())), (g,y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
            debugging_info = cv2.putText(debugging_info, (( "{:3d}, {:3d}").format(a,b)), (g,y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
    return x,i3,g_counting,pc3_counting,debugging_info,details
# key = cv2.waitKey(0)   
import numpy as np


def remove_small_dup(filtered_coord, duplicate_tol):
    #  filtered_segment = sorted_segment_info*(sorted_segment_info[:,2] >debris_filter)[:,None]
    #  filtered_segment = filtered_segment[~np.all(filtered_segment==0, axis =1)]
    #  filtered_segment = np.asarray(sorted(filtered_segment,key=lambda x:x[2]))
     #works to here
     A = np.asarray(sorted(filtered_coord,key=lambda x:x[0]))
     #good to here
     # [C, IA, IC] = uniquetol(filtered_coord,duplicate_tol,"ByRows",True);
     IA = [[1]] * int(len(A[:,0]))
     for j in range(len(A[:,0])):
          for k in range(len(A[:,0])):
               if j != k:
                    IA[j] = IA[j]*((abs(A[j,0]-A[k,0])>duplicate_tol) + (abs(A[j,1]-A[k,1])>duplicate_tol))
     #IA = A[~(np.triu(np.abs(A[:, None] - A) <= duplicate_tol, 1)).any(0)] cant use
     # only IA used
     print(IA)
    #  new_segment_info = filtered_segment * (IA)
    #  new_segment_info = new_segment_info[~np.all(new_segment_info==0, axis =1)]
     return IA

                     
    
## ^^^^^^^^^^^^^^****** ML prepartion ******^^^^^^^^^^^^^^^^ ##
cnn_classifier = load_ML('./checkpoints/1011wbc-pc3.pt')
svm_scorer = pickle.load(open("./checkpoints/oct26.sav", 'rb'))#load svm model

FPS = 59 # 60 frames per second
FLOW_TIME = 1 # second
STOP_PERIOD = 0 # seconds
PROJECTION_TIME = 1 # seconds

FLOW_FRAME = FLOW_TIME * FPS
PAUSE_PACE = (FLOW_TIME + STOP_PERIOD + PROJECTION_TIME) * FPS # FRAMES
INITIAL_PHASE =  (FLOW_TIME + STOP_PERIOD)* FPS
DUPLICATE_TOL = 10

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
VISUALIZE_SIZE = (width, height)
fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
out = cv2.VideoWriter('your_video.avi', fourcc, 20.0, VISUALIZE_SIZE)

try:
    
    print('Starting video. Press CTRL+C to exit.')
    t0 = time.process_time()
    while True:
        start_time = time.process_time()
        check, img = cam.read()# get data and pass them from camera to img
        frame_num  = frame_num + 1   
        if not check:
            # feature_save(details)
            break
        # visualize_video(img)
        data = np.asarray(img)
        GrayFrame = resize_img(data)

        if ReferenceFrame is None:
            ReferenceFrame= GrayFrame
        # img_bw = cc_prc98(GrayFrame)
        img_bw = GrayFrame.copy()
        debugging_info = data.copy() # camera size

        if (frame_num - INITIAL_PHASE) % PAUSE_PACE == 0:
            GrayFrame = process_Frame(ReferenceFrame,GrayFrame)
            end_time = time.process_time()
            print("Reading FPS:", 1 / (end_time - start_time + 1e-6))
            key_reset = cv2.waitKey(1) # WHAT IS THIS
            if key_reset == ord('u'):
                    ReferenceFrame = GrayFrame
                    continue   
            reduced_noise_img = get_reduced_noise_img(ReferenceFrame,data,GrayFrame,BinarizationThreshold)
            #show_full_frame(reduced_noise_img)
            contours, h = cv2.findContours(reduced_noise_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

            x = 0
            i3 = np.zeros((cam_height, cam_width), np.uint8) # camera size
            

            img_bw = cc_prc98(img_bw)
            f_count = 0
            
            centroid = np.empty(len(contours),2)
            for filtered_cell in contours:
                g,y,w,h = cv2.boundingRect(filtered_cell)
                a, b = int((int(g)+int(w/2))), int((int(y) + int(h/2)))   
                centroid[f_count][0] = a
                centroid[f_count][1] = b
                f_count = f_count + 1
            IA = remove_small_dup(centroid, DUPLICATE_TOL)
            c_count = 0
            for truth_value in IA:
                if truth_value == 0:
                    contours[c_count] = None
                c_count = c_count + 1
            
            contours = list(filter(None.__ne__, contours))     
            x,i3,g_counting,pc3_counting,debugging_info,details = [pool.apply(parallel, args=(cell,minContourArea,img_bw,cnn_classifier,frame_num,x,i3,g_counting,pc3_counting,debugging_info,details)) for cell in contours]

            # --------- PROJECTOR -------------------
            projection = np.zeros((cam_height,cam_width), np.uint8) # Creating a dark square with NUMPY camera size
            image = cv2.resize(i3, (projector_width, projector_height))   # Resize frame to projector size
            projection[x_offset:image.shape[0]+x_offset, y_offset:image.shape[1]+y_offset] = image  #Pasting the 'image' into the projector location
            #show_full_frame(projection)
            key = cv2.waitKey(2)
            if key == ord('q'):
                cv2.destroyAllWindows()
            #     break
        if not args.inference:
            imS = cv2.resize(debugging_info, VISUALIZE_SIZE)
            text = "Elapsed Time:" + '{:5.2f}'.format(time.process_time() - t0) + "Video Time: " + '{:5.2f}'.format(frame_num / FPS)
            cv2.putText(imS, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
            cv2.imshow('visualization',imS)
            out.write(imS)      
            # cv2.imshow('visualization',debugging_info)

        # projection = np.zeros((cam_height,cam_width), np.uint8) # Creating a dark square with NUMPY camera size
        # image = cv2.resize(i3, (projector_width, projector_height))   # Resize frame to projector size
        # projection[x_offset:image.shape[0]+x_offset, y_offset:image.shape[1]+y_offset] = image  #Pasting the 'image' into the projector location
        # #show_full_frame(projection)
        # key = cv2.waitKey(2)
        # if key == ord('q'):
        #     cv2.destroyAllWindows()
        #     break
        visualization(t0,data)
        end_time = time.process_time()
        print("TOTAL FPS:", 1/(end_time - start_time + 1e-6))
        print("PC3:", pc3_counting, "total", g_counting)
except KeyboardInterrupt:
    cap.release()
    out.release()
    # cv2.destroyAllWindows()

    cv2.destroyAllWindows()
print('Done.')



