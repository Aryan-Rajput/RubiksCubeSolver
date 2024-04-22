import sys
import cv2
import kociemba
import numpy as np
import time
from scipy import stats
from datetime import datetime


def face_concatenation(upfce, rightfce, frtfce, downfce, leftfce, backface):
    cube_string = ''
    faces = [upfce, rightfce, frtfce, downfce, leftfce, backface]
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
            # hull = cv2.convexHull(contour)
            if cv2.norm(((dper / 4) * (dper / 4)) - A1) < 150:
                # if cv2.ma
                count = count + 1

                # get co ordinates of the contours in the cube
                x, y, w, h = cv2.boundingRect(contour)
                # cv2.rectangle(bgr_image_input, (x, y), (x + w, y + h),
                #               (0, 255, 255), 2)
                # cv2.imshow('cutted contour',
                #            bgr_image_input[y:y + h, x:x + w])
                val = (50 * y) + (10 * x)

                # get mean color of the contour
                color_array = np.array(cv2.mean(
                    bgr_image_input[y:y + h, x:x + w])).astype(int)

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
                
                color_array[0] = hue
                color_array[1] = starn
                color_array[2] = valux

                # print(color_array)
                cv2.drawContours(
                    bgr_image_input, [contour], 0, (255, 255, 0), 2)
                cv2.drawContours(
                    bgr_image_input, [approx], 0, (255, 255, 0), 2)
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
            if (230 <= colors_array[i][0] and
                colors_array[i][1] <= 20 and
                20 <= colors_array[i][2] <= 60):
                colors_array[i][3] = 1
                face[i] = 1
                # print('black detected')
            elif (40 <= colors_array[i][0] <= 80 and
                  60 <= colors_array[i][1] <= 90 and
                  70 <= colors_array[i][2] <= 110):
                colors_array[i][3] = 2
                face[i] = 2
                # print('yellow detected')
            elif (190 <= colors_array[i][0] <= 225 and
                  55 <= colors_array[i][1] <= 95 and
                  35 <= colors_array[i][2] <= 75):
                colors_array[i][3] = 3
                face[i] = 3
                # print('blue detected')
            elif (100 <= colors_array[i][0] <= 150 and
                  25 <= colors_array[i][1] <= 50 and
                  40 <= colors_array[i][2] <= 80):
                colors_array[i][3] = 4
                face[i] = 4
                # print('green detected')
            elif (325 <= colors_array[i][0] <= 365 and
                  50 <= colors_array[i][1] <= 80 and
                  45 <= colors_array[i][2] <= 75):
                colors_array[i][3] = 5
                face[i] = 5
                # print('red detected')
            elif (colors_array[i][0] <= 30 and 
                  65 <= colors_array[i][1] <= 90 and 
                  60 <= colors_array[i][2] <= 90):
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


def rotate_clock_wise(face):
    temp = np.copy(face)
    temp[0, 0], temp[0, 1], temp[0, 2], temp[0, 3], temp[0, 4], temp[0, 5], 
    temp[0, 6], temp[0, 7], temp[0, 8] = face[0, 6], face[0, 3], face[0, 0],
    face[0, 7], face[0, 4], face[0, 1], face[0, 8], face[0, 5], face[0, 2]
    return temp


def rotate_counter_clock_wise(face):
    temp = np.copy(face)
    temp[0, 8], temp[0, 7], temp[0, 6], temp[0, 5], temp[0, 4], temp[0, 3],
    temp[0, 2], temp[0, 1], temp[0, 0] = face[0, 6], face[0, 3], face[0, 0],
    face[0, 7], face[0, 4], face[0, 1], face[0, 8], face[0, 5], face[0, 2]
    return temp


def right_face_clock_wise(video_cap, vid, up_face,
                          right_face, front_face,
                          down_face, left_face, back_face):
    print("Next Move: R Clockwise")
    temp = np.copy(front_face)
    front_face[0, 2] = down_face[0, 2]
    front_face[0, 5] = down_face[0, 5]
    front_face[0, 8] = down_face[0, 8]
    down_face[0, 2] = back_face[0, 6]
    down_face[0, 5] = back_face[0, 3]
    down_face[0, 8] = back_face[0, 0]
    back_face[0, 0] = up_face[0, 8]
    back_face[0, 3] = up_face[0, 5]
    back_face[0, 6] = up_face[0, 2]
    up_face[0, 2] = temp[0, 2]
    up_face[0, 5] = temp[0, 5]
    up_face[0, 8] = temp[0, 8]
    right_face = rotate_clock_wise(right_face)
    # print(right_face)

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video_cap.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        # get current face 
        face, colors_array = face_detection_in_cube(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []

                # if detected face and the actual face after rotaion is same 
                # return the updated faces
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return (up_face, right_face, front_face,
                            down_face, left_face, back_face)

                elif np.array_equal(detected_face, temp) == True:
                    # Get the centroids of the 9th and 3rd color squares
                    centroid1 = colors_array[8]
                    centroid2 = colors_array[2]
                    
                    # Calculate the center points of the two centroids
                    point1 = (centroid1[5]+(centroid1[7]//2),
                              centroid1[6]+(centroid1[7]//2))
                    point2 = (centroid2[5]+(centroid2[8]//2),
                              centroid2[6]+(centroid2[8]//2))
                    
                    # Draw a black arrow from point1 to point2
                    cv2.arrowedLine(bgr_image_input, point1, point2,
                                    (0, 0, 0), 7, tipLength=0.2)
                    
                    # Draw a red arrow over the black 
                    # arrow for better visibility
                    cv2.arrowedLine(bgr_image_input, point1, point2,
                                    (0, 0, 255), 4, tipLength=0.2)
        vid.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break
            

def right_counter_clock_wise(video_cap, vid, up_face, right_face,
                             front_face, down_face, left_face, back_face):
    print("Next Move: R CounterClockwise")
    temp = np.copy(front_face)
    front_face[0, 2] = up_face[0, 2]
    front_face[0, 5] = up_face[0, 5]
    front_face[0, 8] = up_face[0, 8]
    up_face[0, 2] = back_face[0, 6]
    up_face[0, 5] = back_face[0, 3]
    up_face[0, 8] = back_face[0, 0]
    back_face[0, 0] = down_face[0, 8]
    back_face[0, 3] = down_face[0, 5]
    back_face[0, 6] = down_face[0, 2]
    down_face[0, 2] = temp[0, 2]
    down_face[0, 5] = temp[0, 5]
    down_face[0, 8] = temp[0, 8]
    right_face = rotate_counter_clock_wise(right_face)
    # front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video_cap.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, colors_array = face_detection_in_cube(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []

                # if detected face and actual face after
                # rotation is same return the update faces
                if np.array_equal(detected_face, front_face):
                    print("MOVE MADE")
                    return (up_face, right_face, front_face,
                            down_face, left_face, back_face)

                elif np.array_equal(detected_face, temp):
                    # Get the centroids of the 3rd and 9th color squares
                    centroid1 = colors_array[2]
                    centroid2 = colors_array[8]
                    
                    # Calculate the center points of the centroids
                    point1 = (centroid1[5]+(centroid1[7]//2),
                              centroid1[6]+(centroid1[7]//2))
                    point2 = (centroid2[5]+(centroid2[8]//2),
                              centroid2[6]+(centroid2[8]//2))
                    
                    # Draw a black arrow from point1 to point2
                    cv2.arrowedLine(bgr_image_input, point1, point2,
                                    (0, 0, 0), 7, tipLength=0.2)
                    
                    # Draw a red arrow over the black arrow (for visibility)
                    cv2.arrowedLine(bgr_image_input, point1, point2,
                                    (0, 0, 255), 4, tipLength=0.2)
        vid.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break


def left_face_clock_wise(video_cap, vid, up_face, right_face,
                         front_face, down_face, left_face, back_face):
    print("Next Move: L Clockwise")
    temp = np.copy(front_face)
    # Move the left column of the up face to the left column of the front face
    front_face[0, 0] = up_face[0, 0]
    front_face[0, 3] = up_face[0, 3]
    front_face[0, 6] = up_face[0, 6]

    # Move the right column of the back face to
    # the left column of the up face (in reverse order)
    up_face[0, 0] = back_face[0, 8]
    up_face[0, 3] = back_face[0, 5]
    up_face[0, 6] = back_face[0, 2]

    # Move the left column of the down face to 
    # the right column of the back face (in reverse order)
    back_face[0, 2] = down_face[0, 6]
    back_face[0, 5] = down_face[0, 3]
    back_face[0, 8] = down_face[0, 0]

    # Move the left column of the original 
    # front face (stored in temp) to the left column of the down face
    down_face[0, 0] = temp[0, 0]
    down_face[0, 3] = temp[0, 3]
    down_face[0, 6] = temp[0, 6]
    left_face = rotate_clock_wise(left_face)
    # front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video_cap.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, colors_array = face_detection_in_cube(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []

                # if detected face and actual face after
                # rotation is same then return updated faces
                if np.array_equal(detected_face, front_face):
                    print("MOVE MADE")
                    return (up_face, right_face, front_face,
                            down_face, left_face, back_face)
                    
                elif np.array_equal(detected_face, temp):
                    # Get the centroids of the first and last color squares
                    centroid1 = colors_array[0]
                    centroid2 = colors_array[6]
                    
                    # Calculate the center points of
                    # the first and last color squares
                    point1 = (centroid1[5]+(centroid1[7]//2),
                              centroid1[6]+(centroid1[7]//2))
                    point2 = (centroid2[5]+(centroid2[8]//2),
                              centroid2[6]+(centroid2[8]//2))
                    
                    # Draw a black arrow from point1 to point2
                    cv2.arrowedLine(bgr_image_input, point1, point2,
                                    (0, 0, 0), 7, tipLength=0.2)
                    
                    # Draw a red arrow over the black
                    # arrow for better visibility
                    cv2.arrowedLine(bgr_image_input, point1, point2,
                                    (0, 0, 255), 4, tipLength=0.2)
        vid.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break


def left_face_counter_clock_wise(video_cap, vid, up_face, right_face,
                                 front_face, down_face, left_face, back_face):

    print("Next Move: L CounterClockwise")
    temp = np.copy(front_face)
    # Move the left column of the front face to the up face
    front_face[0, 0] = down_face[0, 0]
    front_face[0, 3] = down_face[0, 3]
    front_face[0, 6] = down_face[0, 6]

    # Move the left column of the down face to the back face
    down_face[0, 0] = back_face[0, 8]
    down_face[0, 3] = back_face[0, 5]
    down_face[0, 6] = back_face[0, 2]

    # Move the right column of the back face to the up face
    back_face[0, 2] = up_face[0, 6]
    back_face[0, 5] = up_face[0, 3]
    back_face[0, 8] = up_face[0, 0]

    # Move the left column of the temporary 
    # (original front face) to the up face
    up_face[0, 0] = temp[0, 0]
    up_face[0, 3] = temp[0, 3]
    up_face[0, 6] = temp[0, 6]
    left_face = rotate_counter_clock_wise(left_face)
    # front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video_cap.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, colors_array = face_detection_in_cube(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []

                # if detected face and actual face after the rotation
                # is same then return the updated faces
                if np.array_equal(detected_face, front_face):
                    print("MOVE MADE")
                    return (up_face, right_face, front_face,
                            down_face, left_face, back_face)

                # else display the arrow by calculation 
                # the centroid using the coordinates
                # that are available in color_array
                elif np.array_equal(detected_face, temp):
                    # Calculate the centroid coordinates
                    # for the start and end points of the arrow
                    centroid1 = colors_array[6]
                    centroid2 = colors_array[0]

                    # Calculate the exact points for
                    # the start and end of the arrow
                    point1 = (centroid1[5]+(centroid1[7]//2),
                              centroid1[6]+(centroid1[7]//2))
                    point2 = (centroid2[5]+(centroid2[8]//2),
                              centroid2[6]+(centroid2[8]//2))

                    # Draw a thick black arrow as the outline
                    cv2.arrowedLine(bgr_image_input, point1, point2,
                                    (0, 0, 0), 7, tipLength=0.2)

                    # Draw a thinner red arrow on top of the black outline
                    cv2.arrowedLine(bgr_image_input, point1, point2,
                                    (0, 0, 255), 4, tipLength=0.2)
        vid.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break


def front_face_clock_wise(video_cap, vid, up_face, right_face, front_face,
                         down_face, left_face, back_face):
    print(front_face)
    print("Next Move: F Clockwise")
    temp1 = np.copy(front_face)
    temp = np.copy(up_face)
    front_face = rotate_clock_wise(front_face)
    temp2 = np.copy(front_face)

    # Check if the front face hasn't changed after rotation
    if np.array_equal(temp2, temp1) == True:
        # If the front face is unchanged, perform a series of rotations:

        # 1. Turn the cube to the right
        [up_face, right_face, front_face, 
         down_face, left_face, back_face] = turn_to_right(
             video_cap, vid, up_face, right_face,
             front_face, down_face, left_face, back_face)

        # 2. Perform a clockwise rotation on the left
        # face (which was previously the front face)
        [up_face, right_face, front_face,
         down_face, left_face, back_face] = left_face_clock_wise(
             video_cap, vid, up_face, right_face,
             front_face, down_face, left_face, back_face)
        
        # 3. Turn the cube back to its original front-facing position
        [up_face, right_face, front_face,
         down_face, left_face, back_face] = turn_to_front(
             video_cap, vid, up_face, right_face,
             front_face, down_face, left_face, back_face)
        
        # Return the updated face configurations
        return up_face, right_face, front_face, down_face, left_face, back_face
    
    print("Next Move: F Clockwise")
    temp = np.copy(up_face)
    # Move the up column of the up face to the left face
    up_face[0, 8] = left_face[0, 2]
    up_face[0, 7] = left_face[0, 5]
    up_face[0, 6] = left_face[0, 8]
    # Move the middle column of the left face to the down face
    left_face[0, 2] = down_face[0, 0]
    left_face[0, 5] = down_face[0, 1]
    left_face[0, 8] = down_face[0, 2]
    # Move the bottom row
    down_face[0, 2] = right_face[0, 6]
    down_face[0, 1] = right_face[0, 3]
    down_face[0, 0] = right_face[0, 0]
    # Move the middle column of the right face to the up face
    right_face[0, 0] = temp[0, 0]
    right_face[0, 3] = temp[0, 1]
    right_face[0, 6] = temp[0, 2]
    
    print(front_face)
    faces =  []
    while True:
        is_ok, bgr_image_input = video_cap.read()
        
        if not is_ok:
            print("Cannot read video source")
            sys.exit()
        
        face, colors_array = face_detection_in_cube(bgr_image_input)
        
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []

                # check for current face and actual face that has to be there after the move is made
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return (
                        up_face,
                        right_face,
                        front_face,
                        down_face,
                        left_face,
                        back_face
                    )

                # else display the arrow by calculation the centroid using the coordinates
                # that are available in color_array
                elif np.array_equal(detected_face, temp1) == True:
                    # Calculate centroid points for each corner of the face
                    centroid1 = colors_array[8]
                    centroid2 = colors_array[6]
                    centroid3 = colors_array[0]
                    centroid4 = colors_array[2]

                    # Calculate arrow points based on centroid coordinates
                    # Each point is adjusted to create arrows that form a clockwise rotation
                    point1 = (
                        centroid1[5] + (centroid1[7] // 4),
                        centroid1[6] + (centroid1[7] // 2)
                    )
                    point2 = (
                        centroid2[5] + (3 * centroid2[8] // 4),
                        centroid2[6] + (centroid2[8] // 2)
                    )
                    point3 = (
                        centroid2[5] + (centroid2[7] // 2),
                        centroid2[6] + (centroid2[7] // 4)
                    )
                    point4 = (
                        centroid3[5] + (centroid3[8] // 2),
                        centroid3[6] + (3 * centroid3[8] // 4)
                    )
                    point5 = (
                        centroid3[5] + (3 * centroid3[8] // 4),
                        centroid3[6] + (centroid3[8] // 2)
                    )
                    point6 = (
                        centroid4[5] + (centroid4[8] // 4),
                        centroid4[6] + (centroid4[8] // 2)
                    )
                    point7 = (
                        centroid4[5] + (centroid4[8] // 2),
                        centroid4[6] + (3 * centroid4[8] // 4)
                    )
                    point8 = (
                        centroid1[5] + (centroid1[8] // 2),
                        centroid1[6] + (centroid1[8] // 4)
                    )

                    # Draw black arrows (outline) for better visibility
                    cv2.arrowedLine(
                        bgr_image_input, point1, point2, (0, 0, 0), 7, tipLength=0.2
                    )
                    cv2.arrowedLine(
                        bgr_image_input, point3, point4, (0, 0, 0), 7, tipLength=0.2
                    )
                    cv2.arrowedLine(
                        bgr_image_input, point5, point6, (0, 0, 0), 7, tipLength=0.2
                    )
                    cv2.arrowedLine(
                        bgr_image_input, point7, point8, (0, 0, 0), 7, tipLength=0.2
                    )

                    # Draw red arrows on top of the black arrows
                    cv2.arrowedLine(
                        bgr_image_input, point1, point2, (0, 0, 255), 4, tipLength=0.2
                    )
                    cv2.arrowedLine(
                        bgr_image_input, point3, point4, (0, 0, 255), 4, tipLength=0.2
                    )
                    cv2.arrowedLine(
                        bgr_image_input, point5, point6, (0, 0, 255), 4, tipLength=0.2
                    )
                    cv2.arrowedLine(
                        bgr_image_input, point7, point8, (0, 0, 255), 4, tipLength=0.2
                    )
        vid.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def front_face_counter_clock_wise(video_cap, vid, up_face, right_face, front_face, down_face, left_face, back_face):
    print("Next Move: F CounterClockwise")
    temp = np.copy(up_face)
    temp1 = np.copy(front_face)
    front_face = rotate_counter_clock_wise(front_face)
    temp2 = np.copy(front_face)
    # Check if temp2 and temp1 arrays are equal
    if np.array_equal(temp2, temp1) == True:
        # Turn the cube to the right
        [up_face, right_face, front_face, down_face, left_face, back_face] = turn_to_right(video_cap, vid, up_face, right_face, front_face, down_face, left_face, back_face)
        # Rotate the left face counter-clockwise
        [up_face, right_face, front_face, down_face, left_face, back_face] = left_face_counter_clock_wise(video_cap, vid, up_face, right_face, front_face, down_face, left_face, back_face)
        # Turn the cube back to the front
        [up_face, right_face, front_face, down_face, left_face, back_face] = turn_to_front(video_cap, vid, up_face, right_face, front_face, down_face, left_face, back_face)
        # Return the updated face configurations
        return up_face, right_face, front_face, down_face, left_face, back_face
    
    # Update the top row of the up face with values from the right face
    up_face[0, 6] = right_face[0, 0]
    up_face[0, 7] = right_face[0, 3]
    up_face[0, 8] = right_face[0, 6]

    # Update the first column of the right face with values from the down face
    right_face[0, 0] = down_face[0, 2]
    right_face[0, 3] = down_face[0, 1]
    right_face[0, 6] = down_face[0, 0]

    # Update the top row of the down face with values from the left face
    down_face[0, 0] = left_face[0, 2]
    down_face[0, 1] = left_face[0, 5]
    down_face[0, 2] = left_face[0, 8]

    # Update the last column of the left face with temporary values
    left_face[0, 8] = temp[0, 6]
    left_face[0, 5] = temp[0, 7]
    left_face[0, 2] = temp[0, 8]
    
    print(front_face)
    
    faces = []
    
    while True:
        is_ok, bgr_image_input = video_cap.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, colors_array = face_detection_in_cube(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face, right_face, front_face, down_face, left_face, back_face
                elif np.array_equal(detected_face, temp1) == True:
                    centroid1 = colors_array[2]
                    centroid2 = colors_array[0]
                    centroid3 = colors_array[6]
                    centroid4 = colors_array[8]
                    point1 = (centroid1[5] + (centroid1[7] // 4),
                              centroid1[6] + (centroid1[7] // 2))
                    point2 = (centroid2[5] + (3 * centroid2[8]//4),
                              centroid2[6] + (centroid2[8] // 2))
                    point3 = (centroid2[5] + (centroid2[7] // 2),
                              centroid2[6] + (3 * centroid2[7] // 4))
                    point4 = (centroid3[5] + (centroid3[8] // 2),
                              centroid3[6] + (centroid3[8] // 4))
                    point5 = (centroid3[5] + (3 * centroid3[8] // 4),
                              centroid3[6] + (centroid3[8] // 2))
                    point6 = (centroid4[5] + (centroid4[8] // 4),
                              centroid4[6] + (centroid4[8] // 2))
                    point7 = (centroid4[5] + (centroid4[8] // 2),
                              centroid4[6] + (centroid4[8] // 4))
                    point8 = (centroid1[5] + (centroid1[8] // 2),
                              centroid1[6] + (3 * centroid1[8] // 4))
                    cv2.arrowedLine(bgr_image_input, point1, point2,
                                    (0, 0, 0), 7, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point3, point4,
                                    (0, 0, 0), 7, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point5, point6,
                                    (0, 0, 0), 7, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point7, point8,
                                    (0, 0, 0), 7, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point1, point2,
                                    (0, 0, 255), 4, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point3, point4,
                                    (0, 0, 255), 4, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point5, point6,
                                    (0, 0, 255), 4, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point7, point8,
                                    (0, 0, 255), 4, tipLength=0.2)
        vid.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break


def back_face_clock_wise(video_cap, vid, up_face, right_face, 
                         front_face, down_face, left_face, back_face):
    print("Next Move: B Clockwise")
    temp = np.copy(up_face)

    # Move the up column of the up face to the right face
    up_face[0, 0] = right_face[0, 2]
    up_face[0, 1] = right_face[0, 5]
    up_face[0, 2] = right_face[0, 8]


    # Move the middle column of the right face to the down face
    right_face[0, 8] = down_face[0, 6]
    right_face[0, 5] = down_face[0, 7]
    right_face[0, 2] = down_face[0, 8]

    # move the bottom row of the down face to the left face
    down_face[0, 6] = left_face[0, 0]
    down_face[0, 7] = left_face[0, 3]
    down_face[0, 8] = left_face[0, 6]

    # Move the left column of the temporary (original front face) to the up face
    left_face[0, 0] = temp[0, 2]
    left_face[0, 3] = temp[0, 1]
    left_face[0, 6] = temp[0, 0]
    back_face = rotate_clock_wise(back_face)
    # front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video_cap.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, colors_array = face_detection_in_cube(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return (up_face, right_face, front_face,
                            down_face, left_face, back_face)
                
        vid.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break


def back_face_counter_clock_wise(video_cap, vid, up_face, right_face,
                                 front_face, down_face, left_face, back_face):
    print("Next Move: B CounterClockwise")
    temp = np.copy(up_face)
    # Move the left face to the up face
    up_face[0, 2] = left_face[0, 0]
    up_face[0, 1] = left_face[0, 3]
    up_face[0, 0] = left_face[0, 6]

    # Move the down face to the left face
    left_face[0, 0] = down_face[0, 6]
    left_face[0, 3] = down_face[0, 7]
    left_face[0, 6] = down_face[0, 8]

    # Move the right face to the down face
    down_face[0, 6] = right_face[0, 8]
    down_face[0, 7] = right_face[0, 5]
    down_face[0, 8] = right_face[0, 2]

    # Move the temporary (original up face) to the right face
    right_face[0, 2] = temp[0, 0]
    right_face[0, 5] = temp[0, 1]
    right_face[0, 8] = temp[0, 2]
    back_face = rotate_counter_clock_wise(back_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video_cap.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, colors_array = face_detection_in_cube(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return (up_face, right_face, front_face,
                            down_face, left_face, back_face)
        vid.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break


def up_face_clock_wise(video_cap, vid, up_face, right_face,
                       front_face, down_face, left_face, back_face):
    print("Next Move: U Clockwise")
    temp = np.copy(front_face)
    # Move the top row of the right face to the front face
    front_face[0, 0] = right_face[0, 0]
    front_face[0, 1] = right_face[0, 1]
    front_face[0, 2] = right_face[0, 2]

    # Move the top row of the back face to the right face
    right_face[0, 0] = back_face[0, 0]
    right_face[0, 1] = back_face[0, 1]
    right_face[0, 2] = back_face[0, 2]

    # Move the top row of the left face to the back face
    back_face[0, 0] = left_face[0, 0]
    back_face[0, 1] = left_face[0, 1]
    back_face[0, 2] = left_face[0, 2]

    # Move the top row of the temporary (original front face) to the left face
    left_face[0, 0] = temp[0, 0]
    left_face[0, 1] = temp[0, 1]
    left_face[0, 2] = temp[0, 2]
    up_face = rotate_clock_wise(up_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video_cap.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, colors_array = face_detection_in_cube(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return (up_face, right_face, front_face,
                            down_face, left_face, back_face)
                elif np.array_equal(detected_face, temp) == True:
                    # Get the centroids of two specific 
                    # colors from the colors_array
                    centroid1 = colors_array[2]
                    centroid2 = colors_array[0]

                    # Calculate the start and end points for the arrow
                    # The arrow starts from the center of centroid1 and ends at the center of centroid2
                    point1 = (centroid1[5]+(centroid1[7]//2),
                              centroid1[6]+(centroid1[7]//2))
                    point2 = (centroid2[5]+(centroid2[8]//2),
                              centroid2[6]+(centroid2[8]//2))

                    # Draw a thick black arrow on the input image
                    cv2.arrowedLine(bgr_image_input, point1, point2,
                                    (0, 0, 0), 7, tipLength=0.2)

                    # Draw a thinner red arrow on top of 
                    # the black arrow for better visibility
                    cv2.arrowedLine(bgr_image_input, point1, point2,
                                    (0, 0, 255), 4, tipLength=0.2)
        vid.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break
            

def up_face_counter_clock_wise(video_cap, vid, up_face, right_face, 
                               front_face, down_face, left_face, back_face):
    print("Next Move: U CounterClockwise")
    temp = np.copy(front_face)
    # Move the top row of the left face to the front face
    front_face[0, 0] = left_face[0, 0]
    front_face[0, 1] = left_face[0, 1]
    front_face[0, 2] = left_face[0, 2]

    # Move the top row of the back face to the left face
    left_face[0, 0] = back_face[0, 0]
    left_face[0, 1] = back_face[0, 1]
    left_face[0, 2] = back_face[0, 2]

    # Move the top row of the right face to the back face
    back_face[0, 0] = right_face[0, 0]
    back_face[0, 1] = right_face[0, 1]
    back_face[0, 2] = right_face[0, 2]

    # Move the top row of the temporary (original front face) to the right face
    right_face[0, 0] = temp[0, 0]
    right_face[0, 1] = temp[0, 1]
    right_face[0, 2] = temp[0, 2]
    up_face = rotate_counter_clock_wise(up_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video_cap.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, colors_array = face_detection_in_cube(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return (up_face, right_face, front_face,
                            down_face, left_face, back_face)
                elif np.array_equal(detected_face, temp) == True:
                    # Get the centroids of the first and third color arrays
                    centroid1 = colors_array[0]
                    centroid2 = colors_array[2]

                    # Calculate the start and end points for the arrow
                    # The arrow starts from the center of the first centroid
                    point1 = (centroid1[5]+(centroid1[7]//2),
                              centroid1[6]+(centroid1[7]//2))
                    # The arrow ends at the center of the third centroid
                    point2 = (centroid2[5]+(centroid2[8]//2),
                              centroid2[6]+(centroid2[8]//2))

                    # Draw a thick black arrow on the image
                    cv2.arrowedLine(bgr_image_input, point1, point2,
                                    (0, 0, 0), 7, tipLength=0.2)
                    # Draw a thinner red arrow on top of the black arrow
                    cv2.arrowedLine(bgr_image_input, point1, point2,
                                    (0, 0, 255), 4, tipLength=0.2)
        vid.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break
            

def down_face_clock_wise(video_cap, vid, up_face, right_face,
                         front_face, down_face, left_face, back_face):
    print("Next Move: D Clockwise")
    temp = np.copy(front_face)
    # Move the bottom row of the front face to the left face
    front_face[0, 6] = left_face[0, 6]
    front_face[0, 7] = left_face[0, 7]
    front_face[0, 8] = left_face[0, 8]

    # Move the bottom row of the left face to the back face
    left_face[0, 6] = back_face[0, 6]
    left_face[0, 7] = back_face[0, 7]
    left_face[0, 8] = back_face[0, 8]

    # Move the bottom row of the back face to the right face
    back_face[0, 6] = right_face[0, 6]
    back_face[0, 7] = right_face[0, 7]
    back_face[0, 8] = right_face[0, 8]

    # Move the bottom row of the right face to the
    # front face (using the temporary variable)
    right_face[0, 6] = temp[0, 6]
    right_face[0, 7] = temp[0, 7]
    right_face[0, 8] = temp[0, 8]

    # These moves collectively rotate the bottom layer of the cube clockwise
    down_face = rotate_clock_wise(down_face)
    # front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video_cap.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, colors_array = face_detection_in_cube(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return (up_face, right_face, front_face,
                            down_face, left_face, back_face)
                elif np.array_equal(detected_face, temp) == True:
                    # Calculate the center points of two specific color squares
                    centroid1 = colors_array[6]
                    centroid2 = colors_array[8]

                    # Calculate the coordinates for the start and end points of the arrow
                    point1 = (centroid1[5]+(centroid1[7]//2),
                              centroid1[6]+(centroid1[7]//2))
                    point2 = (centroid2[5]+(centroid2[8]//2),
                              centroid2[6]+(centroid2[8]//2))

                    # Draw a black arrow (outline) on the input image
                    cv2.arrowedLine(bgr_image_input, point1, point2,
                                    (0, 0, 0), 7, tipLength = 0.2)

                    # Draw a red arrow (inner line) on top of the black arrow
                    cv2.arrowedLine(bgr_image_input, point1, point2,
                                    (0, 0, 255), 4, tipLength=0.2)
        vid.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break
            

def down_face_counter_clock_wise(video_cap, vid, up_face, right_face,
                  front_face, down_face, left_face, back_face):
    print("Next Move: D CounterClockwise")
    temp = np.copy(front_face)
    # Move the bottom row of the right face to the front face
    front_face[0, 6] = right_face[0, 6]
    front_face[0, 7] = right_face[0, 7]
    front_face[0, 8] = right_face[0, 8]

    # Move the bottom row of the back face to the right face
    right_face[0, 6] = back_face[0, 6]
    right_face[0, 7] = back_face[0, 7]
    right_face[0, 8] = back_face[0, 8]

    # Move the bottom row of the left face to the back face
    back_face[0, 6] = left_face[0, 6]
    back_face[0, 7] = left_face[0, 7]
    back_face[0, 8] = left_face[0, 8]

    # Move the bottom row of the temporary (original front face) to the left face
    left_face[0, 6] = temp[0, 6]
    left_face[0, 7] = temp[0, 7]
    left_face[0, 8] = temp[0, 8]
    down_face = rotate_counter_clock_wise(down_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video_cap.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, colors_array = face_detection_in_cube(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return (up_face, right_face, front_face,
                            down_face, left_face, back_face)
                elif np.array_equal(detected_face,temp) == True:
                    # Calculate the center points of two specific color regions
                    centroid1 = colors_array[8]
                    centroid2 = colors_array[6]

                    # Calculate the start and end points for the arrow
                    point1 = (centroid1[5]+(centroid1[7]//2), centroid1[6]+(centroid1[7]//2))
                    point2 = (centroid2[5]+(centroid2[8]//2), centroid2[6]+(centroid2[8]//2))

                    # Draw a black arrow (outline) on the image
                    cv2.arrowedLine(bgr_image_input, point1, point2,
                                    (0, 0, 0), 7, tipLength = 0.2)

                    # Draw a red arrow (inner line) on top of the black arrow
                    cv2.arrowedLine(bgr_image_input, point1, point2,
                                    (0, 0, 255), 4, tipLength=0.2)
        vid.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break


def turn_to_right(video_cap, vid, up_face, right_face,
                  front_face, down_face, left_face, back_face):
    print("Next Move: Show Right Face")
    temp = np.copy(front_face)
    front_face = np.copy(right_face)
    right_face = np.copy(back_face)
    back_face = np.copy(left_face)
    left_face = np.copy(temp)
    up_face = rotate_clock_wise(up_face)
    down_face = rotate_counter_clock_wise(down_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video_cap.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, colors_array = face_detection_in_cube(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return (up_face, right_face, front_face,
                            down_face, left_face, back_face)
                elif np.array_equal(detected_face, temp) == True:
                    # Extract centroids from colors_array
                    centroid1 = colors_array[8]
                    centroid2 = colors_array[6]
                    centroid3 = colors_array[5]
                    centroid4 = colors_array[3]
                    centroid5 = colors_array[2]
                    centroid6 = colors_array[0]

                    # Calculate midpoints for each centroid
                    point1 = (centroid1[5] + (centroid1[7] // 2),
                              centroid1[6] + (centroid1[7] // 2))
                    point2 = (centroid2[5] + (centroid2[8] // 2),
                              centroid2[6] + (centroid2[8] // 2))
                    point3 = (centroid3[5] + (centroid3[7] // 2),
                              centroid3[6] + (centroid3[7] // 2))
                    point4 = (centroid4[5] + (centroid4[8] // 2),
                              centroid4[6] + (centroid4[8] // 2))
                    point5 = (centroid5[5] + (centroid5[7] // 2),
                              centroid5[6] + (centroid5[7] // 2))
                    point6 = (centroid6[5] + (centroid6[8] // 2),
                              centroid6[6] + (centroid6[8] // 2))

                    # Draw black arrows (thicker lines)
                    cv2.arrowedLine(bgr_image_input, point1, point2,
                                    (0, 0, 0), 7, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point3, point4,
                                    (0, 0, 0), 7, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point5, point6,
                                    (0, 0, 0), 7, tipLength=0.2)

                    # Draw red arrows (thinner lines) on top of black arrows
                    cv2.arrowedLine(bgr_image_input, point1, point2,
                                    (0, 0, 255), 4, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point3, point4,
                                    (0, 0, 255), 4, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point5, point6,
                                    (0, 0, 255), 4, tipLength=0.2)

        vid.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

            

def turn_to_front(video_cap, vid, up_face, right_face,
                  front_face, down_face, left_face, back_face):
    print("Next Move: Show Front Face")
    temp = np.copy(front_face)
    front_face = np.copy(left_face)
    left_face = np.copy(back_face)
    back_face = np.copy(right_face)
    right_face = np.copy(temp)
    up_face = rotate_counter_clock_wise(up_face)
    down_face = rotate_clock_wise(down_face)
    #front_face = temp
    
    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video_cap.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, colors_array = face_detection_in_cube(bgr_image_input)

        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                detected_face = stats.mode(face_array)[0]
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                
                if np.array_equal(detected_face, front_face):
                    print("Move Made")
                    return (up_face, right_face, front_face,
                            down_face, left_face, back_face)
                else:
                    # Calculate the center points for the arrows
                    centroid1 = colors_array[6]
                    centroid2 = colors_array[8]
                    centroid3 = colors_array[3]
                    centroid4 = colors_array[5]
                    centroid5 = colors_array[0]
                    centroid6 = colors_array[2]

                    # Calculate the start and end points for each arrow
                    # Each point is calculated by adding half of 
                    # the width/height to the x/y coordinates
                    point1 = (centroid1[5] + (centroid1[7] // 2),
                              centroid1[6] + (centroid1[7] // 2))
                    point2 = (centroid2[5] + (centroid2[8] // 2),
                              centroid2[6] + (centroid2[8] // 2))
                    point3 = (centroid3[5] + (centroid3[7] // 2),
                              centroid3[6] + (centroid3[7] // 2))
                    point4 = (centroid4[5] + (centroid4[8] // 2),
                              centroid4[6] + (centroid4[8] // 2))
                    point5 = (centroid5[5] + (centroid5[7] // 2),
                              centroid5[6] + (centroid5[7] // 2))
                    point6 = (centroid6[5] + (centroid6[8] // 2),
                              centroid6[6] + (centroid6[8] // 2))

                    # Draw black arrows (outer lines) on the image
                    cv2.arrowedLine(bgr_image_input, point1, point2,
                                    (0, 0, 0), 7, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point3, point4,
                                    (0, 0, 0), 7, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point5, point6,
                                    (0, 0, 0), 7, tipLength=0.2)

                    # Draw red arrows (inner lines) on top of the black arrows
                    cv2.arrowedLine(bgr_image_input, point1, point2,
                                    (0, 0, 255), 4, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point3, point4,
                                    (0, 0, 255), 4, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point5, point6,
                                    (0, 0, 255), 4, tipLength=0.2)

        vid.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

    return None
        





def find_face_in_cube(video_cap, vid, uf, rf, ff, df, lf, bf, text=""):
    faces = []
    while True:
        tr, bgr_image_input = video_cap.read()

        if not tr:
            print("Cannot read video source")
            sys.exit()
        # assinging values to face and blob colors 
        # based on the face_detection_in_cube method
        face, clr_arr = face_detection_in_cube(bgr_image_input)
        bgr_image_input = cv2.putText(
            bgr_image_input, text, (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
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
                if (np.array_equal(detected_face, uf) == False and
                    np.array_equal(detected_face, ff) == False and
                    np.array_equal(detected_face, bf) == False and
                    np.array_equal(detected_face, df) == False and
                    np.array_equal(detected_face, lf) == False and
                    np.array_equal(detected_face, rf) == False):
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
        print(f"Failed to open video_cap writer: {e}")
        sys.exit(1)
    
    while True:
        tr, bgr_image_input = video_cap.read()
        if not tr:
            break
        while True:
            # Capture frame-by-frame
            # for frtfce
            frtfce = find_face_in_cube(video_cap, vid, upfce, rightfce,
                                       frtfce, downfce, leftfce, backfce,
                                       text="Show Front Face")
            mf = frtfce[0, 4]
            print(frtfce)
            print(type(frtfce))
            print(mf)
            
            # -> upfce change
            upfce = find_face_in_cube(video_cap, vid, upfce, rightfce,
                                      frtfce, downfce, leftfce, backfce,
                                      text="Show Top Face")
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
                    bgr_image_input = cv2.putText(
                        bgr_image_input, "Show Down Face",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    vid.write(bgr_image_input)
                    cv2.imshow("Output Image", bgr_image_input)
                    key_pressed = cv2.waitKey(1) & 0xFF
                    if key_pressed == 27 or key_pressed == ord('q'):
                        broke = 1
                        break
            if broke == 1:
                break
            mu = upfce[0, 4]
            print(upfce)
            print(mu)
            
            # -> Down face change
            downfce = find_face_in_cube(video_cap, vid, upfce, rightfce,
                                        frtfce, downfce, leftfce, backfce,
                                        text="Show Down Face")
            start_time = datetime.now()
            while True:

                if (datetime.now() - start_time).total_seconds() > 3:
                    break
                else:
                    tr, bgr_image_input = video_cap.read()
                    if not tr:
                        broke = 1
                        break
                    bgr_image_input = cv2.putText(
                        bgr_image_input, "Show Right Face",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
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
            rightfce = find_face_in_cube(video_cap, vid, upfce, rightfce,
                                         frtfce, downfce, leftfce, backfce,
                                         text="Show Right Face")
            start_time = datetime.now()
            while True:
                if (datetime.now() - start_time).total_seconds() > 3:
                    break
                else:
                    tr, bgr_image_input = video_cap.read()
                    if not tr:
                        broke = 1
                        break
                    bgr_image_input = cv2.putText(
                        bgr_image_input, "Show Left Face",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
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
            leftfce = find_face_in_cube(video_cap, vid, upfce, rightfce,
                                        frtfce, downfce, leftfce, backfce,
                                        text="Show Left Face")
            start_time = datetime.now()
            while True:
                if (datetime.now() - start_time).total_seconds() > 3:
                    break
                else:
                    tr, bgr_image_input = video_cap.read()
                    if not tr:
                        broke = 1
                        break
                    bgr_image_input = cv2.putText(
                        bgr_image_input, "Show Back Face",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
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
            backfce = find_face_in_cube(video_cap, vid, upfce, rightfce,
                                        frtfce, downfce, leftfce, backfce,
                                        text="Show Back Face")
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
                is_ok, bgr_image_input = video_cap.read()
                bgr_image_input = cv2.putText(bgr_image_input, "CUBE ALREADY SOLVED", (100, 50),
                                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                vid.write(bgr_image_input)
                cv2.imshow("Output Image", bgr_image_input)
                key_pressed = cv2.waitKey(1) & 0xFF
                if key_pressed == 27 or key_pressed == ord('q'):
                    break
                time.sleep(5)
                break
            # assigning respective values to the faces
            ''' F -------> Front face
                R -------> Right face
                B -------> Back face
                L -------> Left face
                U -------> Up face
                D -------> Down face'''
            final_string = ''
            for val in range(len(solution)):
                if solution[val] == mf:
                    final_string = final_string + 'F'
                elif solution[val] == mr:
                    final_string = final_string + 'R'
                elif solution[val] == mb:
                    final_string = final_string + 'B'
                elif solution[val] == ml:
                    final_string = final_string + 'L'
                elif solution[val] == mu:
                    final_string = final_string + 'U'
                elif solution[val] == md:
                    final_string = final_string + 'D'

            print(final_string)
            try:
                solved = kociemba.solve(final_string)
                print(solved)
                break
            except :
                upfce = [0, 0]
                frtfce = [0, 0]
                leftfce = [0, 0]
                rightfce = [0, 0]
                downfce = [0, 0]
                backfce = [0, 0]

        if broke == 1:
            break
        # spliting the steps and calling respective functions so that arrows can be displayed accordingly
        # below methods are available in the rotate.py file
        # in steps letter like R F B D L U indicate right, front, back , down faces to be rotated clockwise
        # if there is 2 after the letter they should be rotate twice and if there is " ' " like R' then the respective
        # face should be rotated anti clock wise
        steps = solved.split()
        for step in steps:
            if step == "R":
                [upfce, rightfce, frtfce, downfce, leftfce, backfce] = right_face_clock_wise(
                    video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce
                )
            elif step == "R'":
                [upfce, rightfce, frtfce, downfce, leftfce, backfce] = right_counter_clock_wise(
                    video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce
                )
            elif step == "R2":
                [upfce, rightfce, frtfce, downfce, leftfce, backfce] = right_face_clock_wise(
                    video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce
                )
                [upfce, rightfce, frtfce, downfce, leftfce, backfce] = right_face_clock_wise(
                    video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce
                )
            elif step == "L":
                [upfce, rightfce, frtfce, downfce, leftfce, backfce] = left_face_clock_wise(
                    video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce
                )
            elif step == "L'":
                [upfce, rightfce, frtfce, downfce, leftfce, backfce] = left_face_counter_clock_wise(
                    video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce
                )
            elif step == "L2":
                [upfce, rightfce, frtfce, downfce, leftfce, backfce] = left_face_clock_wise(
                    video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce
                )
                [upfce, rightfce, frtfce, downfce, leftfce, backfce] = left_face_clock_wise(
                    video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce
                )
            elif step == "F":
                [upfce, rightfce, frtfce, downfce, leftfce, backfce] = front_face_clock_wise(
                    video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce
                )
            elif step == "F'":
                [upfce, rightfce, frtfce, downfce, leftfce, backfce] = front_face_counter_clock_wise(
                    video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce
                )
            elif step == "F2":
                [upfce, rightfce, frtfce, downfce, leftfce, backfce] = front_face_clock_wise(
                    video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce
                )
                [upfce, rightfce, frtfce, downfce, leftfce, backfce] = front_face_clock_wise(
                    video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce
                )
            elif step == "B":
                [upfce, rightfce, frtfce, downfce, leftfce, backfce] = turn_to_right(
                    video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce
                )
                [upfce, rightfce, frtfce, downfce, leftfce, backfce] = right_face_clock_wise(
                    video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce
                )
                [upfce, rightfce, frtfce, downfce, leftfce, backfce] = turn_to_front(
                    video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce
                )
            elif step == "B'":
                [upfce, rightfce, frtfce, downfce, leftfce, backfce] = turn_to_right(
                    video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce
                )
                [upfce, rightfce, frtfce, downfce, leftfce, backfce] = right_counter_clock_wise(
                    video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce
                )
                [upfce, rightfce, frtfce, downfce, leftfce, backfce] = turn_to_front(
                    video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce
                )
            elif step == "B2":
                [upfce, rightfce, frtfce, downfce, leftfce, backfce] = turn_to_right(
                    video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce
                )
                [upfce, rightfce, frtfce, downfce, leftfce, backfce] = right_face_clock_wise(
                    video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce
                )
                [upfce, rightfce, frtfce, downfce, leftfce, backfce] = right_face_clock_wise(
                    video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce
                )
                [upfce, rightfce, frtfce, downfce, leftfce, backfce] = turn_to_front(
                    video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce
                )
            elif step == "U":
                [upfce, rightfce, frtfce, downfce, leftfce, backfce] = up_face_clock_wise(
                    video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce
                )
            elif step == "U'":
                [upfce, rightfce, frtfce, downfce, leftfce, backfce] = up_face_counter_clock_wise(
                    video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce
                )
            elif step == "U2":
                [upfce, rightfce, frtfce, downfce, leftfce, backfce] = up_face_clock_wise(
                    video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce
                )
                [upfce, rightfce, frtfce, downfce, leftfce, backfce] = up_face_clock_wise(
                    video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce
                )
            elif step == "D":
                [upfce, rightfce, frtfce, downfce, leftfce, backfce] = down_face_clock_wise(
                    video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce
                )
            elif step == "D'":
                [upfce, rightfce, frtfce, downfce, leftfce, backfce] = down_face_counter_clock_wise(
                    video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce
                )
            elif step == "D2":
                [upfce, rightfce, frtfce, downfce, leftfce, backfce] = down_face_clock_wise(
                    video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce
                )
                [upfce, rightfce, frtfce, downfce, leftfce, backfce] = down_face_clock_wise(
                    video_cap, vid, upfce, rightfce, frtfce, downfce, leftfce, backfce
                )
        while True:
            # Capture frame-by-frame
            ret, frame = video_cap.read()
            # Display the resulting frame
            cv2.imshow('Video', frame)
            # Press 'q' to quit
    
    # When everything is done, release the capture
    
    
    