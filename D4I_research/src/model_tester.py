import cv2
import numpy as np
import glob
import random
import time
import csv
# Load Yolo
import timeit
start1 = timeit.default_timer()
net = cv2.dnn.readNet("../checkpoints/yolov3_training_1000.weights", '../checkpoints/yolov3_testing.cfg') 

# Name custom object
classes = ["Cell"]
details = []
# Images path
## DISCLAIMER: TODO: DYNAMIC INPUT PROCESSING; NEED TO MAKE IT SO USER INPUT IS PROCESSED IN "GLOB" RATHER THAN HARD CODE
images_path = glob.glob(r"../data/videos/check/*.tif")

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
conf_threshold = 0.4
nms_threshold = 0.4
start2 = timeit.default_timer()
def coordinate_save (details):
    all_data = []
    labels = ['a','b', 'frame', 'type','w','h']
    all_data = [labels]
    all_data.extend(details)
    filename = "dl_coordinate.csv"
    with open("../data/features/"+filename, "w", encoding="utf-8") as f:
        f_csv = csv.writer(f)
        f_csv.writerows(all_data)
        f.close()
# Insert here the path of your images
random.shuffle(images_path)


# loop through all the images
for img_path in images_path:
    # Loading image
    # img = (cv2.imread(img_path)/256).astype('uint8')
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    maximum_gray = np.amax(img)
    minimum_gray = np.amin(img)
    alpha = 255/(maximum_gray - minimum_gray)
    beta = - minimum_gray * alpha
    # img = cv2.convertScaleAbs(img,alpha = alpha, beta = beta)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    t0 = time.time()
    outs = net.forward(output_layers)
    t = time.time()
    print('inference time: ', t-t0)


    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                print(confidence)
                # Object detected
                print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                a,b,t,u = x/0.4,y/0.4,w/0.4,h/0.4
                #a, b = int((int(a)+int(t/2))), int((int(b) + int(u/2)))
                coordinates = "x: {} y: {} w: {} h: {}".format(a, b, t, u)
                print(coordinates)
                print("\n")

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                details.append([a,b,240,class_id,t,u])
    coordinate_save(details)
    # indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    # print(indexes)
    # font = cv2.FONT_HERSHEY_PLAIN
    print(range(len(boxes)))
    for i in range(len(boxes)):
        # if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
stop = timeit.default_timer()
print('yolo Time: ', stop - start1) 
print('yolo Time: ', stop - start2) 
# COLLAB MENCHANISM
# %matplotlib inline

import matplotlib.pyplot as plt
# load image using cv2....and do processing.
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# as opencv loads in BGR format by default, we want to show it in RGB.
plt.show()


#     cv2.imshow(img)
#     key = cv2.waitKey(0)

# cv2.destroyAllWindows()