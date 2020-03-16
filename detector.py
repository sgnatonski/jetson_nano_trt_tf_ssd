import sys
import time
import argparse
import traceback
import sys
import os
import queue
import threading
import multiprocessing
import json
import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver
import numpy as np

from utils.ssd_classes import get_cls_dict
from utils.ssd import TrtSSD
from utils.visualization import BBoxVisualization
from utils.display import open_window, set_display, show_fps

parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
'''
Command line options
'''
parser.add_argument(
    '--rtsp_cam', 	type=str, help='RTSP camera address', default=os.environ.get('RTSP_CAM', None)
)
parser.add_argument(
    '--ui', 		type=int, help='should use CV2 for detection rendering', default=os.environ.get('RTSP_CAM_UI', None)
)

def initVC():
    print("Connecting video stream")
    model_config = parser.parse_args()
    cap = cv2.VideoCapture(model_config.rtsp_cam)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
    return cap

def cap_reconnect(cap):
    faults = 0
    while(cap is None or cap.isOpened() == False):
        cap = initVC()
        print("Back off for " + str(.5 * faults) + "s")
        time.sleep(.5 * faults)
        if faults < 20:
            faults += 1
    return cap

def get_frame(q):
    cap = None
    try:
        cap = cap_reconnect(None)
    except Exception as e:
        traceback.print_exc()
        cap = cap_reconnect(cap)

    while(cap.isOpened()):
        while q.full():
            cap.grab()
        try:
            ret, frame = cap.read()
            if frame is None:
                raise Exception("Received empty frame") 
            else:
                q.put((frame, time.time()))	
        except Exception as e:
            traceback.print_exc()
            cap = cap_reconnect(None)    

def detect(q1, q2):      
    model_config = parser.parse_args()
    print("Loading detector")
    model = "ssd_mobilenet_v2_coco"
    conf_th = 0.3
    INPUT_HW = (300, 300)
    cls_dict = get_cls_dict("coco")
    vis = BBoxVisualization(cls_dict)
    trt_ssd = TrtSSD(model, INPUT_HW)
    print("Loading detector complete")
    if model_config.ui == 1:
        cv2.startWindowThread()
        cv2.namedWindow("window")

    while 1:
        try:
            frame, frame_time = q1.get()
            delay = time.time() - frame_time
            if delay > 0.4:
                print("Skipping frame")
                continue       
            boxes, confs, clss = trt_ssd.detect(frame, conf_th)
            print([get_cls_dict("coco")[c] for c in clss])
            if model_config.ui == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = vis.draw_bboxes(frame, boxes, confs, clss)
                cv2.imshow('window', img[..., ::-1])
        except Exception as e:
            traceback.print_exc()

q1 = multiprocessing.Queue(maxsize=1)
q2 = multiprocessing.Queue(maxsize=1)

if __name__ == '__main__':
    
    t1 = multiprocessing.Process(target=get_frame, args = (q1,))
    t1.daemon = True
    
    t2 = multiprocessing.Process(target=detect, args = (q1,q2))
    t2.daemon = True

    t1.start()
    t2.start()
    t2.join()
    t1.join()
