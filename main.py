import sys
import os
import cv2
import kociemba
import numpy as np
import time
from scipy import stats
from datetime import datetime


def face_concatenation(up_face, right_face, front_face, down_face, left_face, back_face):
    cube_string = ''
    faces = [up_face, right_face, front_face, down_face, left_face, back_face]
    for face in faces:
        for row in face:
            for color in row:
                cube_string += color
    return cube_string


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
    gray = cv2.adaptiveThreshold(
        gray, 37, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 0
    )
    # cv2.imshow('gray',gray)
    try:
        _, contours, hierarchy = cv2.findContours(
            gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
    except:
        contours, hierarchy = cv2.findContours(
            gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
    i = 0
    contour_id = 0
    # print(len(contours))
    count = 0
    colors_array = []
    for contour in contours:
        # get area of contours
        A1 = cv2.contourArea(contour)
        contour_id = contour_id + 1

        if A1 < 3000 and A1 > 1000:
            dper = cv2.arcLength(contour, True)

            eps = 0.01 * dper
            approx = cv2.approxPolyDP(contour, eps, True)
            # this is a just in case scenario
            hull = cv2.convexHull(contour)
            if cv2.norm(((dper / 4) * (dper / 4)) - A1) < 150:
                # if cv2.ma
                count = count + 1

                # get co ordinates of the contours in the cube
                x, y, w, h = cv2.boundingRect(contour)
                # cv2.rectangle(bgr_image_input, (x, y), (x + w, y + h), (0, 255, 255), 2)
                # cv2.imshow('cutted contour', bgr_image_input[y:y + h, x:x + w])
                val = (50 * y) + (10 * x)

                # get mean color of the contour
                color_array = np.array(cv2.mean(bgr_image_input[y:y + h, x:x + w])).astype(int)

                blue = color_array[0] / 255
                green = color_array[1] / 255
                red = color_array[2] / 255

                cmax = max(red, blue, green)
                cmin = min(red, blue, green)
                diff = cmax - cmin
                hue = -1
                starn = -1

                if (cmax == cmin):
                    hue = 0

                elif (cmax == red):
                    hue = (60 * ((green - blue) / diff) + 360) % 360

                elif (cmax == green):
                    hue = (60 * ((blue - red) / diff) + 120) % 360

                elif (cmax == blue):
                    hue = (60 * ((red - green) / diff) + 240) % 360

                if (cmax == 0):
                    starn = 0
                else:
                    starn = (diff / cmax) * 100

                valux = cmax * 100

                # print(hue,starn,valux)
                # exit()
                # print(color_array)
                # print(hue,starn,valux),valux)
                # exit()
                # print(color_array)r_array)
                # print(hue,starne,starn
                
                color_array[0], color_array[1], color_array[2] = hue, saturation, value

                # print(color_array)
                cv2.drawContours(bgr_image_input, [contour], 0, (255, 255, 0), 2)
                cv2.drawContours(bgr_image_input, [approx], 0, (255, 255, 0), 2)
                color_array = np.append(color_array, val)
                color_array = np.append(color_array, x)
                color_array = np.append(color_array, y)
                color_array = np.append(color_array, w)
                color_array = np.append(color_array, h)
                colors_array.append(color_array)
    if len(colors_array) > 0:
        colors_array = np.asarray(colors_array)
        colors_array = colors_array[colors_array[:, 4].argsort()]
    face = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    if len(colors_array) == 9:
        # print(colors_array)
        for i in range(9):
            if 230 <= colors_array[i][0] and colors_array[i][1] <= 20 and 20 <= colors_array[i][2] <= 60:
                colors_array[i][3] = 1
                face[i] = 1
                # print('black detected')
            elif 40 <= colors_array[i][0] <= 80 and 60 <= colors_array[i][1] <= 90 and 70 <= colors_array[i][2] <= 110:
                colors_array[i][3] = 2
                face[i] = 2
                # print('yellow detected')
            elif 190 <= colors_array[i][0] <= 225 and 55 <= colors_array[i][1] <= 95 and 35 <= colors_array[i][2] <= 75:
                colors_array[i][3] = 3
                face[i] = 3
                # print('blue detected')
            elif 100 <= colors_array[i][0] <= 150 and 25 <= colors_array[i][1] <= 50 and 40 <= colors_array[i][2] <= 80:
                colors_array[i][3] = 4
                face[i] = 4
                # print('green detected')
            elif 325 <= colors_array[i][0] <= 365 and 50 <= colors_array[i][1] <= 80 and 45 <= colors_array[i][2] <= 75:
                colors_array[i][3] = 5
                face[i] = 5
                # print('red detected')
            elif colors_array[i][0] <= 30 and 65 <= colors_array[i][1] <= 90 and 60 <= colors_array[i][2] <= 90:
                colors_array[i][3] = 6
                face[i] = 6
                # print('orange detected')
        if np.count_nonzero(face) == 9:
            return face, colors_array
        else:
            return [0, 0], colors_array
    else:
        return [0, 0, 0], colors_array
        # break


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
    upfce = [0, 0]
    downfce = [0, 0]
    frtfce = [0, 0]
    backfce = [0, 0]
    leftfce = [0, 0]
    rightfce = [0, 0]
    
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
    except Exception as e:
        print(f"Failed to open video writer: {e}")
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
            mu = up_face[0, 4]
            print(up_face)
            print(mu)
            
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
            #using kociembaa module to solve the cube
            solution = face_concatenation(upfce, rightfce, frtfce, downfce, leftfce, backfce)
            # print(solution)
            cube_solved = [mu, mu, mu, mu, mu, mu, mu, mu, mu, mr, mr, mr, mr, mr, mr, mr, mr, mr, mf, mf, mf, mf, mf,
                           mf, mf, mf, mf, md, md, md, md, md, md, md, md, md, ml, ml, ml, ml, ml, ml, ml, ml, ml, mb,
                           mb, mb, mb, mb, mb, mb, mb, mb]
            if (face_concatenation(upfce, rightfce, frtfce, downfce, leftfce, backfce) == cube_solved).all():
                is_ok, bgr_image_input = video.read()
                bgr_image_input = cv2.putText(bgr_image_input, "CUBE ALREADY SOLVED", (100, 50),
                                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                vid.write(bgr_image_input)
                cv2.imshow("Output Image", bgr_image_input)
                key_pressed = cv2.waitKey(1) & 0xFF
                if key_pressed == 27 or key_pressed == ord('q'):
                    break
                time.sleep(5)
                break
        
        while True:
            # Capture frame-by-frame
            ret, frame = video_cap.read()
            # Display the resulting frame
            cv2.imshow('Video', frame)
            # Press 'q' to quit
    
    # When everything is done, release the capture
    
    
    