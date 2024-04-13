import sys
import os
import cv2
import kociemba
import numpy as np
import time
from scipy import stats
from datetime import datetime


def main():
    upfce = [0,0]
    downfce = [0,0]
    frontfce = [0,0]
    backfce = [0,0]
    leftfce = [0,0]
    rightfce = [0,0]
    
    video_capture = cv2.VideoCapture(0)
    is_ok, bgr_frame = video_capture.read()
    broke = 0
    
    # cv2.imshow('Video', bgr_frame)
    if not is_ok:
        print("Failed to capture image")
        sys.exit(1)
    
    h1 = bgr_frame.shape[0]
    w1 = bgr_frame.shape[1]
    faces = []
    
    try:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fname = "OUTPUTS.avi"
        fps = 20
        out = cv2.VideoWriter(fname, fourcc, fps, (w1, h1))
    except:
        print("Failed to open video writer")
        sys.exit(1)
    
    while True:
        # Capture frame-by-frame
        # for front_face
        frontfce = find_face(bgr_frame, frontfce, text = "front")
        mf = frontfce[0,4]
        print(frontfce)
        print(type(frontfce))
        print(mf)
        
        # for down_face
        downfce = find_face(bgr_frame, downfce, text = "down")
        start_time = datetime.now()
        # this loop is to stay same side for 3 seconds so that by mistake wrong color will not be detected
        # due to lighting factor
        while True:
            if(datetime.now() - start_time).seconds > 3:
                break
            else:
                is_ok, bgr_frame = video_capture.read()
                if not is_ok:
                    print("Failed to capture image")
                    sys.exit(1)
                bgr_frame = cv2.putText(bgr_frame, "Please stay in same side for 3 seconds", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                downfce = find_face(bgr_frame, downfce, text = "down")
        
        upfce = find_face(bgr_frame, upfce, text = "up")
        backfce = find_face(bgr_frame, backfce, text = "back")
        leftfce = find_face(bgr_frame, leftfce, text = "left")
        rightfce = find_face(bgr_frame, rightfce, text = "right")
    
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        # Display the resulting frame
        cv2.imshow('Video', frame)
        # Press 'q' to quit
        
    # When everything is done, release the capture