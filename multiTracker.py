# Copyright 2018 BIG VISION LLC ALL RIGHTS RESERVED
# 
from __future__ import print_function
import sys
import cv2
from random import randint
import video_demo
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from util import *
from darknet import Darknet
import  pandas as pd
import pickle


trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]: 
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)

    return tracker
# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])



if __name__ == '__main__':


    ####################################
    # detection codes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = {
        "video": "video.avi", # Video to run detection upon
        "dataset": "pasacal", # Dataset on which the network has been trained
        "confidence": 0.5, # Object Confidence to filter predictions
        "nms_thresh": 0.4, # NMS Threshold
        "cfgfile": "cfg/yolov3.cfg", # Config file
        "weightsfile": "yolov3.weights", # Weightsfile
        "repo": 416, # Input resolution of the network.  Increase to increase accuracy.  Decrease to increase speed
        "iou_thresh": 0.1
    }

    confidence = float(params["confidence"])
    nms_thresh = float(params["nms_thresh"])

    num_classes = 80
    bbox_attr = 5 + num_classes

    bboxes = []
    xywh = []
    (H, W) = (None, None) 
    counter = 0
    line = [(43, 543), (550, 655)]


    model = Darknet(params["cfgfile"])
    model.load_weights(params["weightsfile"])
    model.net_info["height"] = params["repo"]
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    model.to(device)
    model.eval()

    #videofile = params["video"]
    #cap = cv2.VideoCapture(0)

    ##########################################



    #print("Default tracking algoritm is CSRT \n"
    #      "Available tracking algorithms are:\n")

    for t in trackerTypes:
        print(t)

    trackerType = "CSRT"

    # Set video to load
    video_path = "videos/small_ex01.avi"

    # Create a video capture object to read videos
    cap = cv2.VideoCapture(video_path)

    # Read first frame
    success, frame = cap.read()
    # quit if unable to read the video file
    if not success:
        #print('Failed to read video')
        sys.exit(1)
        
    ## Select boxes
    #bboxes = video_demo.demo()
    colors = []
    #for i in bboxes:
    #    colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))


    ## Initialize MultiTracker
    # There are two ways you can initialize multitracker
    # 1. tracker = cv2.MultiTracker("CSRT")
    # All the trackers added to this multitracker
    # will use CSRT algorithm as default
    # 2. tracker = cv2.MultiTracker()
    # No default algorithm specified

    # Initialize MultiTracker with tracking algo
    # Specify tracker type

    # Create MultiTracker object
    multiTracker = cv2.MultiTracker_create()

    # Initialize MultiTracker 
    #for bbox in bboxes:
    #    print("tracking input box: ", bbox)
    #    multiTracker.add(createTrackerByName(trackerType), frame, bbox)


    frames = 0
    previous_boxes = []


    # Process video and track objects
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
          break
        det_boxes = []

        ##########################
        # detection

        # detection phase
        if frames == 0:
            img, orig_im, dim = video_demo.prep_image(frame, inp_dim)
            im_dim = torch.FloatTensor(dim).repeat(1,2)
            im_dim = im_dim.to(device)
            img = img.to(device)

            with torch.no_grad():
                output = model(Variable(img), torch.cuda.is_available())
            output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thresh)

            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)

            output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
            output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2

            output[:,1:5] /= scaling_factor
            if W is None or H is None:
                (H, W) = frame.shape[:2]

            for i in range(output.shape[0]):
                output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
                output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])

            #print("output: ", output)
            #print("output: ", output.shape)
            classes = load_classes('data/coco.names')

            for i in output:
                x0 = int(i[1].item())
                y0 = int(i[2].item())
                x1 = int(i[3].item())
                y1 = int(i[4].item())
                cls = int(i[-1].item())
                label = "{0}".format(classes[cls])
                #bbox = (x0, y0, x1, y1)
                #det_boxes.append(bbox)
                # print(box.shape)
                w = x1 - x0
                h = y1 - y0
                if label in ["person", "car"]:
                # if label == "person":
                    det_boxes.append((x0, y0, w, h))
                #print(det_boxes)

            # Create MultiTracker object
            multiTracker = cv2.MultiTracker_create()

            # Initialize MultiTracker 
            for bbox in det_boxes:
                # colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
                #print("tracking input box: ", bbox)
                multiTracker.add(createTrackerByName(trackerType), frame, bbox)

        ##########################

        # tracking phase
        # get updated location of objects in subsequent frames
        success, trc_boxes = multiTracker.update(frame)
        trc_boxes = [[int(i) for i in box] for box in trc_boxes]
        print("-------------")
        # print("tracked boxes: ", trc_boxes)

        # Compare the detection boxes and tracking boxes
        if frames == 0:
            det_boxes = [list(box) for box in det_boxes]
            # print("detected boxes: ", det_boxes)
            boxes = compare_boxes(det_boxes, trc_boxes, params['iou_thresh'])
            # print("boxes:", boxes)
            if frames % 15 == 0 or (frames+1) % 15 == 0:
                print("detected boxes: ", det_boxes)
                print("boxes:", boxes)

        # draw tracked objects
        for i, newbox in enumerate(boxes):
            # print("  i: ", i)
            # print("  newbox: ", newbox)
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            previous_p1 = ()
            previous_p2 = ()


            # color = (randint(64, 255), randint(64, 255), randint(64, 255))
            color = (255, 0, 0)
            cv2.rectangle(frame, p1, p2, color, 2, 1)
            #print("rectangle point: ", p1, p2)
            cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
            
            if previous_boxes != ():
                if len(previous_boxes) > i:
                    n_center = (int(newbox[0] + newbox[2] / 2),
                                int(newbox[1] + newbox[3] / 2))
                    p_center = (int(previous_boxes[i][0] + previous_boxes[i][2] / 2),
                                int(previous_boxes[i][1] + previous_boxes[i][3] / 2))
                    if intersect(p_center, n_center, (0, H // 2), (W, H // 2)):
                        counter += 1

        # if frames % 15 == 0 and frames != 0:
        #     # print(len(det_boxes), len(boxes))
        #     counter -= int(len(boxes) / 2)
        print(frames, counter)
        previous_boxes = boxes
        # draw counter
        cv2.putText(frame, str(counter), (100,200), cv2.FONT_HERSHEY_DUPLEX, 5.0, (0, 255, 255), 10)
        #counter += 1
        cv2.imshow('MultiTracker', frame)

        frames += 1
        #print("frames>>>>>", frames)

        # quit on ESC button
        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            break
