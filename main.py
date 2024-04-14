import sys
import os
import cv2
import kociemba
import numpy as np
import time
from scipy import stats
from datetime import datetime


def face_detection_in_cube(bgr_image_input):
    # convert  image to gray
    gray = cv2.cvtColor(bgr_image_input, cv2.COLOR_BGR2GRAY)

    # defining kerel for morphological operations using ellipse structure
    krl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, krl)

    # cv2.imshow('gray',gray)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, krl)
    # gray = cv2.Canny(bgr_image_input,50,100)
    # cv2.imshow('gray',gray)
    gray = cv2.adaptiveThreshold(gray, 37, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 0)
    # cv2.imshow('gray',gray)
    
    # Find contours in the image
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mnar = 1000
    mxar = 10000
    re_contours = [cnt for cnt in contours if mnar < cv2.contourArea(cnt) < mxar]
    srt_contours = sorted(re_contours, key=cv2.contourArea, reverse=True)
    cb_faces = srt_contours[:9]
    clr_arr = []
    for cnt in cb_faces:
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [cnt], 0, 255, -1)
        mean_color = cv2.mean(bgr_image_input, mask=mask)[:3]
        clr_arr.append(mean_color)
    
    cv2.drawContours(bgr_image_input, cb_faces, -1, (0, 255, 0), 2)
    
    return bgr_image_input, clr_arr
    

def find_face_in_cube(video_cap, vid, uf, rf, ff, df, lf, bf, text=""):
    faces = []
    while True:
        tr, bgr_image_input = video_cap.read()

        if not tr:
            print("Cannot read video source")
            sys.exit()
        # assinging values to face and blob colors based on the face_detection_in_cube method
        face, clr_arr = face_detection_in_cube(bgr_image_input)
        bgr_image_input = cv2.putText(bgr_image_input, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 5:
                face_array = np.array(faces)
                # print('asd')
                detected_face = stats.mode(face_array)[0]
                # print(final_face)
                uf = np.asarray(uf)
                ff = np.asarray(ff)
                detected_face = np.asarray(detected_face)
                # print(np.array_equal(detected_face, tf))
                # print(np.array_equal(detected_face, ff))
                
                faces = []
                if (np.array_equal(detected_face, uf) == False and np.array_equal(detected_face, ff) == False and 
                    np.array_equal(detected_face, bf) == False and np.array_equal(detected_face, df) == False and 
                    np.array_equal(detected_face, lf) == False and np.array_equal(detected_face, rf) == False):
                    return detected_face
        vid.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break



def main():
    upfce = [0,0]
    downfce = [0,0]
    frtfce = [0,0]
    backfce = [0,0]
    leftfce = [0,0]
    rightfce = [0,0]
    
    video_cap = cv2.VideoCapture(0)
    tr, bgr_frame = video_cap.read()
    broke = 0
    
    # cv2.imshow('Video', bgr_frame)
    if not tr:
        print("Failed to capture image")
        sys.exit(1)
    
    h1 = bgr_frame.shape[0]
    w1 = bgr_frame.shape[1]
    faces = []
    
    try:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fname = "OUTPUTS.avi"
        fps = 20
        vid = cv2.VideoWriter(fname, fourcc, fps, (w1, h1))
    except:
        print("Failed to open video writer")
        sys.exit(1)
    
    while True:
        tr, bgr_image_input = video_cap.read()
        if not tr:
            break
        while True:
            # Capture frame-by-frame
            # for frtfce
            frtfce = find_face_in_cube(video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce, text="Show Front Face")
            mf = frtfce[0, 4]
            print(frtfce)
            print(type(frtfce))
            print(mf)
            
            
            # -> upfce change
            upfce = find_face_in_cube(video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce, text="Show Top Face")
            start_time = datetime.now()
            # -> this loop is to stay same side for 3 seconds so that by 
            # mistake wrong color will not be detected
            # -> due to lighting factor
            while True:
                if (datetime.now() - start_time).total_seconds() > 3:
                    break
                else:
                    tr, bgr_image_input = video_cap.read()
                    if not tr:
                        broke = 1
                        break
                    bgr_image_input = cv2.putText(bgr_image_input, "Show Down Face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                                  2, (0, 0, 255), 3)
                    vid.write(bgr_image_input)
                    cv2.imshow("Output Image", bgr_image_input)
                    key_pressed = cv2.waitKey(1) & 0xFF
                    if key_pressed == 27 or key_pressed == ord('q'):
                        broke = 1
                        break
            if broke == 1:
                break
            
            # -> Down face change
            downfce = find_face_in_cube(video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce, text="Show Down Face")
            start_time = datetime.now()
            while True:

                if (datetime.now() - start_time).total_seconds() > 3:
                    break
                else:
                    tr, bgr_image_input = video_cap.read()
                    if not tr:
                        broke = 1
                        break
                    bgr_image_input = cv2.putText(bgr_image_input, "Show Right Face", (50, 50),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    vid.write(bgr_image_input)
                    cv2.imshow("Output Image", bgr_image_input)
                    key_pressed = cv2.waitKey(1) & 0xFF
                    if key_pressed == 27 or key_pressed == ord('q'):
                        broke = 1
                        break
            if broke == 1:
                break
            md = downfce[0, 4]
            print(downfce)
            print(md)
            
            # Right Face Change
            rightfce = find_face_in_cube(video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce, text="Show Right Face")
            start_time = datetime.now()
            while True:
                if (datetime.now() - start_time).total_seconds() > 3:
                    break
                else:
                    tr, bgr_image_input = video_cap.read()
                    if not tr:
                        broke = 1
                        break
                    bgr_image_input = cv2.putText(bgr_image_input, "Show Left Face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                                  2, (0, 0, 255), 3)
                    vid.write(bgr_image_input)
                    cv2.imshow("Output Image", bgr_image_input)
                    key_pressed = cv2.waitKey(1) & 0xFF
                    if key_pressed == 27 or key_pressed == ord('q'):
                        broke = 1
                        break
            if broke == 1:
                break
            mr = rightfce[0, 4]
            print(rightfce)
            print(mr)
            
            # Left Face Change 
            leftfce = find_face_in_cube(video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce, text="Show Left Face")
            start_time = datetime.now()
            while True:
                if (datetime.now() - start_time).total_seconds() > 3:
                    break
                else:
                    tr, bgr_image_input = video_cap.read()
                    if not tr:
                        broke = 1
                        break
                    bgr_image_input = cv2.putText(bgr_image_input, "Show Back Face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                                  2, (0, 0, 255), 3)
                    vid.write(bgr_image_input)
                    cv2.imshow("Output Image", bgr_image_input)
                    key_pressed = cv2.waitKey(1) & 0xFF
                    if key_pressed == 27 or key_pressed == ord('q'):
                        broke = 1
                        break
            if broke == 1:
                break
            ml = leftfce[0, 4]
            print(leftfce)
            print(ml)
            
            
            # Back Face Change
            backfce = find_face_in_cube(video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce, text="Show Back Face")
            start_time = datetime.now()
            while True:
                if (datetime.now() - start_time).total_seconds() > 3:
                    break
                else:
                    tr, bgr_image_input = video_cap.read()
                    if not tr:
                        broke = 1
                        break
                    bgr_image_input = cv2.putText(bgr_image_input, "Show Front Face", (50, 50),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    vid.write(bgr_image_input)
                    cv2.imshow("Output Image", bgr_image_input)
                    key_pressed = cv2.waitKey(1) & 0xFF
                    if key_pressed == 27 or key_pressed == ord('q'):
                        broke = 1
                        break
            if broke == 1:
                break
            mb = backfce[0, 4]
            print(backfce)
            # time.sleep(2)
            print(mb)
            
        
        while True:
            # Capture frame-by-frame
            ret, frame = video_cap.read()
            # Display the resulting frame
            cv2.imshow('Video', frame)
            # Press 'q' to quit
    
    # When everything is done, release the capture