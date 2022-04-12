##
# Face mask detector - application to find faces without a mask and create an alert if someone doesn't have one.
##

import cv2
import sys
import numpy as np
from math import sqrt
from gtts import gTTS
import pygame
import datetime

from config import *

##
# Resizes frame based on the given factor.
##
def frame_resize(frame, rescale_factor):
    width = int(frame.shape[1] * rescale_factor / 100)
    height = int(frame.shape[0] * rescale_factor / 100)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    return frame


##
# Finds faces in the given frame.
##
def find_faces(frame, face_detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray)

    return faces


##
# Checks if the mask is present on the part of the face
# based on the given threshold.
##
def calculate_partial_mask_presence(color_distance, threshold):
    if color_distance >= threshold:
        return True
    else:
        return False


##
# Checks if the mask is present on the face.
##
def calculate_mask_result(distance_nose, distance_whole):
    threshold_nose = THRESHOLD_NOSE
    threshold_whole = THRESHOLD_WHOLE

    mask_nose_present = calculate_partial_mask_presence(distance_nose, threshold_nose)
    mask_whole_present = calculate_partial_mask_presence(distance_whole, threshold_whole)

    if (mask_nose_present and mask_whole_present):
        mask_result = True
    else:
        mask_result = False

    return mask_result


##
# Calculates proportions of the face.
##
def calculate_parts(face, height_part, width_part):
    (x, y, w, h) = face

    part_height = int(h / height_part)
    part_width = int(w / width_part)

    return part_height, part_width

##
# Plays an information audio, that the mask should be worn.
##
def text_to_speech(mask_result, tts_cooldown, no_mask):
    if not mask_result and no_mask > 0:
        no_mask -= 1
    elif mask_result:
        no_mask = 10
    if tts_cooldown == 0 and not mask_result and no_mask == 0:
        pygame.mixer.music.play()
        print(f"Wykryto brak maseczki: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        tts_cooldown = 40

    return tts_cooldown, no_mask


##
# Displays info for calibration and debugging purposes.
##
def display_calibration(frame, avg_color_upper_face, avg_color_lower_face_nose, avg_color_lower_face_whole,
                        distance_nose, distance_whole):
    cv2.rectangle(frame, (40, 10), (70, 40), (int(avg_color_upper_face[0]),
                                              int(avg_color_upper_face[1]),
                                              int(avg_color_upper_face[2])), -1)

    cv2.rectangle(frame, (100, 10), (130, 40), (int(avg_color_lower_face_nose[0]),
                                                int(avg_color_lower_face_nose[1]),
                                                int(avg_color_lower_face_nose[2])), -1)

    cv2.rectangle(frame, (160, 10), (190, 40), (int(avg_color_lower_face_whole[0]),
                                                int(avg_color_lower_face_whole[1]),
                                                int(avg_color_lower_face_whole[2])), -1)

    cv2.putText(frame,
                f"Distance nose: {distance_nose}", \
                (40, 70), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.putText(frame,
                f"Distance whole: {distance_whole}", \
                (40, 100), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


##
# Analyses face to check if the mask is present.
##
def analyze_face(frame, face, height_part, width_part):
    (x, y, w, h) = face

    # Constants for calculating used pixel ranges.
    part_height, part_width = calculate_parts(face, height_part, width_part)

    cv2.rectangle(frame, (x + part_width, y), (x + 4 * part_width, y + part_height), (255, 0, 0), 2)

    # Upper part of the face.
    x_left_upper_face = x + part_width
    x_right_upper_face = x + 4 * part_width + 1
    y_up_upper_face = y
    y_down_upper_face = y + part_height + 1

    avg_color_upper_face = calculate_average_color_range(frame,
                                                         x_left_upper_face,
                                                         x_right_upper_face,
                                                         y_up_upper_face,
                                                         y_down_upper_face)

    ## Calculating nose.
    x_left_lower_face = x + part_width
    x_right_lower_face = x + 4 * part_width + 1
    lower_face_parts = 3
    lower_face_part_range = int(part_height / lower_face_parts)
    y_up_lower_face_nose = y + part_height
    y_down_lower_face_nose = y + part_height + lower_face_part_range + 1

    avg_color_lower_face_nose = calculate_average_color_range(frame,
                                                              x_left_lower_face,
                                                              x_right_lower_face,
                                                              y_up_lower_face_nose,
                                                              y_down_lower_face_nose)

    ## Calculating whole lower face.
    y_up_lower_face_whole = y + part_height
    y_down_lower_face_whole = y + h + 1

    avg_color_lower_face_whole = calculate_average_color_range(frame,
                                                               x_left_lower_face,
                                                               x_right_lower_face,
                                                               y_up_lower_face_whole,
                                                               y_down_lower_face_whole)

    distance_nose, distance_whole = calculate_distances(avg_color_upper_face,
                                                        avg_color_lower_face_nose,
                                                        avg_color_lower_face_whole)

    mask_result= calculate_mask_result(distance_nose, distance_whole)

    ## Calibration
    if CALIBRATION_MODE:
        display_calibration(frame, avg_color_upper_face,
                            avg_color_lower_face_nose, avg_color_lower_face_whole,
                            distance_nose, distance_whole)

    return mask_result


##
# Calculates rectangle color for lower face, depending
# on presence of the mask.
##
def calculate_rectangle_color(mask_result):
    if mask_result:
        return (0, 255, 0)
    else:
        return (0, 0, 255)


##
# Displays info about presence of the mask.
##
def display_result(frame, face, height_part, width_part, mask_result):
    (x, y, w, h) = face

    part_height, part_width = calculate_parts(face, height_part, width_part)

    # Upper part of the face.
    x_left_upper_face = x + part_width
    x_right_upper_face = x + 4 * part_width + 1
    y_up_upper_face = y
    y_down_upper_face = y + part_height + 1

    ## Calculating nose.
    x_left_lower_face = x + part_width
    x_right_lower_face = x + 4 * part_width + 1
    lower_face_parts = 3
    lower_face_part_range = int(part_height / lower_face_parts)
    y_up_lower_face_nose = y + part_height
    y_down_lower_face_nose = y + part_height + lower_face_part_range + 1

    ## Calculating whole lower face.
    y_up_lower_face_whole = y + part_height
    y_down_lower_face_whole = y + h + 1

    whole_rec_color = calculate_rectangle_color(mask_result)

    mask_info = "No mask!"

    if mask_result:
        mask_info = "Mask is present!"

    cv2.rectangle(frame,
                  (x_left_lower_face, y_down_lower_face_whole),
                  (x_right_lower_face, y_up_lower_face_whole),
                  whole_rec_color, 2)

    cv2.putText(frame,
                f"Mask presence: {mask_info}", \
                (x + 30, y - 10), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


##
# Calculates color distance.
##
def color_distance(color):
    return sqrt(color[0] ** 2 + color[1] ** 2 + color[2] ** 2)


##
# Calculates average color in the given frame range.
##
def calculate_average_color_range(calculated_frame, x_left, x_right, y_up, y_down):
    avg_color = np.zeros(3)
    count = 0

    for x_pixel in range(x_left, x_right):
        for y_pixel in range(y_up, y_down):
            try:
                avg_color += calculated_frame[y_pixel, x_pixel]
                count += 1
            except IndexError:
                pass

    if count > 0:
        avg_color = (avg_color / count).astype(int)

    return avg_color


##
# Calculates color distances between parts of a face.
##
def calculate_distances(avg_color_upper_face, avg_color_lower_face_whole,
                        avg_color_lower_face_nose):
    distance_whole = int(abs(color_distance(avg_color_upper_face) - color_distance(avg_color_lower_face_whole)))
    distance_nose = int(abs(color_distance(avg_color_upper_face) - color_distance(avg_color_lower_face_nose)))

    return distance_whole, distance_nose


##
# Detects masks on faces.
##
def face_mask_detect():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    face_detector = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    rescale_factor = 100

    height_part = 2
    width_part = 5

    tts_cooldown = 20
    no_mask = 10

    # Read until video is completed
    while (cap.isOpened()):

        everyone_mask = True

        if tts_cooldown > 0:
            tts_cooldown -= 1

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            frame = frame_resize(frame, rescale_factor)
            faces = find_faces(frame, face_detector)

            for face in faces:
                mask_result = analyze_face(frame, face, height_part, width_part)
                display_result(frame, face, height_part, width_part, mask_result)

                if mask_result == False:
                    everyone_mask = False

            tts_cooldown, no_mask = text_to_speech(everyone_mask, tts_cooldown, no_mask)

            cv2.imshow('Wideo', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


## Text to speech settings.
language = 'pl'
tts = gTTS(text="Proszę założyć maseczkę", lang=language, slow=False)
tts.save("announcement.mp3")
pygame.mixer.init()
pygame.mixer.music.load("announcement.mp3")

## Running the program.
face_mask_detect()
