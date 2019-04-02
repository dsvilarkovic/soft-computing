import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy


def green_mask(frame):
    dummy_frame = frame.copy()
    
    hsv = cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2HSV)
    
    lower_green = (40,40,40)
    upper_green = (70, 255, 255)
    
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    #res = cv2.bitwise_and(dummy_frame, dummy_frame, mask = mask)
    
    return 255 - mask

def red_mask(frame):
    dummy_frame = frame.copy()
    
    hsv = cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2HSV)
    
    lower_red = (10,50,20)
    upper_red = (179, 255, 255)
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    #res = cv2.bitwise_and(dummy_frame, dummy_frame, mask = mask)
    
    return 255 - mask

vid = cv2.VideoCapture('video-0.avi')

while vid.isOpened():
    flag, number_frame = vid.read()

    mask_g = green_mask(number_frame)
    process_frame = cv2.bitwise_and(number_frame, number_frame, mask = mask_g)
    mask_r = red_mask(process_frame)
    digit_frame = cv2.bitwise_and(process_frame, process_frame, mask = mask_r)

    
    kernel = np.ones((2,2),np.uint8)
    digit_frame = cv2.erode(digit_frame, kernel, iterations = 1)
    digit_frame = cv2.dilate(digit_frame, kernel, iterations = 2)

    cv2.imshow('Slika', digit_frame)

    if cv2.waitKey(1) & 0xFF == 27 or flag==False:
        break

cv2.destroyAllWindows()
vid.release()     




