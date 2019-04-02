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
    
    res = cv2.bitwise_and(dummy_frame, dummy_frame, mask = mask)
    
    return 255 - mask

def red_mask(frame):
    dummy_frame = frame.copy()
    
    hsv = cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2HSV)
    
    lower_red = (10,50,20)
    upper_red = (179, 255, 255)
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    res = cv2.bitwise_and(dummy_frame, dummy_frame, mask = mask)
    
    return 255 - mask


#cleans numbers from possible red or green color
#and convert to grayscale if wanter
def extract_numbers(frame, gray = False):
    mask_g = green_mask(frame)
    no_green_frame = cv2.bitwise_and(frame, frame, mask = mask_g)
    mask_r = red_mask(no_green_frame)
    res = cv2.bitwise_and(no_green_frame, no_green_frame, mask = mask_r)
    
    if(gray):
        res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
    
    return res

def find_boxes(digit_frame):
    gray = cv2.cvtColor(digit_frame, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    binary = cv2.dilate(binary, (5, 5), iterations=3)

    (im2, contours, hierarchy) = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for c in contours:
    #filter out the noise
        if cv2.contourArea(c) < 20:
            continue
                
        #get bounding box 28x28 from countour        
        (x, y, w, h) = cv2.boundingRect(c)
        x = x - int((28 - w)/2)
        y = y - int((28 - h)/2)
        w = 28
        h = 28
        bboxes.append([x, y, w, h])
            
    return bboxes


def main():
    vid = cv2.VideoCapture('video-0.avi')

    while vid.isOpened():
        flag, number_frame = vid.read()
        digit_frame = extract_numbers(number_frame)
        bboxes = find_boxes(digit_frame)



        

        for box in bboxes:
            (x,y,w,h) = box
            cv2.rectangle(digit_frame, (x,y), (x + w, y + h), color=(0, 255, 0))
            
        cv2.imshow('Slika', digit_frame)


        if cv2.waitKey(1) & 0xFF == 27 or flag==False:
            break

    cv2.destroyAllWindows()
    vid.release()     



main()