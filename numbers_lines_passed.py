import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
import random
import math as math
from keras.models import load_model
from keras.models import model_from_json

mnist_model = load_model('model_1.h5')
      


histories = []

def predict(image):
    return np.argmax(mnist_model.predict([image]))

def green_mask(frame):
    dummy_frame = frame.copy()
    hsv = cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2HSV)
    
    lower_green = (40,40,40)
    upper_green = (70, 255, 255)
    
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    
    return 255 - mask

def red_mask(frame):
    dummy_frame = frame.copy()
    
    
    hsv = cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2HSV)
   


    lower_red = (10,50,20)
    upper_red = (179, 255, 255)
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    return 255 - mask


def parted_hough_line_unifier(lines):
    """Finds whole line consisted of parts"""
    #print(lines)
    point_begin = []
    point_end = []
    point_begin.append(lines[0][0][0])
    point_begin.append(lines[0][0][1])
    point_end.append(lines[0][0][2])
    point_end.append(lines[0][0][3])
    
    
    for i in range(lines.shape[0]):
        for x1,y1,x2,y2 in lines[i]:
            print(x1, y1, x2,y2)
            
            
    for line in lines[1:]:
        for x1,y1,x2,y2 in lines[i]:
            if(x1 < point_begin[0] ):
        #check if point_begin is legit, if not change it
                point_begin[0] = x1
                point_begin[1] = y1
            
        #check if point_end is legit, if not change it
            if(x2 > point_end[0]):
                point_end[0] = x2
                point_end[1] = y2

            
    return (point_begin[0], point_begin[1], point_end[0], point_end[1])
        

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

    (_, contours, _) = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

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

def additional_cleaning(digit_frame):
    img = cv2.cvtColor(digit_frame.copy(), cv2.COLOR_RGB2GRAY) 
    binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    (_, contours, _) = cv2.findContours(binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if(len(contours) == 0):
        return digit_frame
    
    
    bboxes = []

    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        bboxes.append([x, y, w, h])

   
    (x, y, w, h) = bboxes[0]

    refurbish_img = np.uint8(np.zeros((28,28,3)))
    position_x = int((28 - w)/2)
    position_y = int((28 - h)/2)            
    refurbish_img[position_y:position_y + h,position_x:position_x + w,:] = digit_frame[y:y+h,x:x+w, :]
    return refurbish_img


#keep only one color channel
def process_line(frame, color_channel):
    frame_color = frame.copy()
    frame_color = frame_color[:,:,color_channel]
    
    
    kernel = np.ones((3,3),np.uint8)
    frame_color = cv2.erode(frame_color, kernel, iterations = 1)
    frame = cv2.dilate(frame_color, kernel, iterations = 1)
    
    return frame_color

#function returns only one line 
def hough_lines_return(frame):
    gray = frame
    gray = np.where(gray < 10, 0, gray).astype(np.uint8)
    edges = cv2.Canny(gray, threshold1 = 50, threshold2= 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold = 100, maxLineGap = 10 )

    
    (x1,y1,x2,y2) = parted_hough_line_unifier(lines)  #lines[0,0,:]
    
    return (x1,y1,x2,y2)

BLUE = 0
GREEN = 1
EPSILON_Y = 5
EPSILON_D = 20

class LineSegment:
    def __init__(self, x1,y1,x2,y2):
        self.point_begin = (x1,y1)
        self.point_end = (x2,y2)

        self.getLineEquation(x1,y1,x2,y2)


    def getLineEquation(self, x1,y1,x2,y2):
        self.m = (y2 - y1)/float(x2 - x1)
        self.c = y1 - self.m * x1

        return (self.m, self.c)

    def getLineCrossed(self, x, y):
        check1 = y - self.m*x -  self.c < 0
        return check1


    def getProjectionPoint(self, point):
        """Returns projection point on line segment"""
        return None

    def getShortestDistance(self, point):
        a = self.m 
        b = -1
        c = self.c

        x1 = point[0]
        y1 = point[1]

        #need distance to see how far from line is point
        distance = abs((a * x1 + b * y1 + c ) / (math.sqrt(a * a + b * b)))

        #need exact point on line that is perpendicular to point 

        return distance

    def pointIsClose(self, point):

        x = point[0]
        y = point[1]

        
        x1 = self.point_begin[0]
        y1 = self.point_begin[1]

        x2 = self.point_end[0]
        y2 = self.point_end[1]

        
        if(x1 <= x <= x2 and y2 - EPSILON_Y <= y <= y1 + EPSILON_Y):
            return True

        return False

    def pointIsBelowLine(self, point):
        x = point[0]
        y = point[1]

        y_made = self.m * x + self.c

        if(y < y_made - EPSILON_Y):
            return True

        return False
    def drawLine(self, cv2, frame, color_chosen = (0,255,0)):
        cv2.line(frame, self.point_begin, self.point_end, color = color_chosen , thickness = 3)
        #cv2.rectangle(frame, self.point_begin- EPSILON, self.point_end - EPSILON, color = (0, 255,0))
        x1 = self.point_begin[0]
        y1 = self.point_begin[1]

        x2 = self.point_end[0]
        y2 = self.point_end[1]

        cv2.putText(frame, 'begin(' + str(x1) + ', ' + str(y1) + ')', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        cv2.putText(frame, 'end(' + str(x2) + ', ' + str(y2) + ')', (x2,y2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

def eucledianDistance(a, b):
    x1 = a[0] - b[0]
    x2 = a[1] - b[1]
    return np.linalg.norm([x1, x2])


def findAndPredictNumberOnLine(frame, bbox, found_index, lineSegment, intersections, vid):
    """Returns tuple (boolean ifIntersect, int prediction) \n
        ifIntersect - returns True if bbox intersects line \n
        prediction - return predicted number found in the box"""
    
    global histories
    if(len(histories[found_index]) <= 10):
        return (False, -1)
    
    (x, y, w, h) = bbox
    centroid_x = x - 14
    centroid_y = y - 14

    centroid = (centroid_x, centroid_y)
    d = lineSegment.getShortestDistance(centroid)
    y = lineSegment.m * centroid_x + lineSegment.c
    if(d < EPSILON_D and (found_index not in intersections.keys())):
        if(lineSegment.pointIsClose(centroid)):


            intersections[found_index] = centroid
            print('Intersection REALLY found!')

            (old_centroid_x, old_centroid_y) = histories[found_index][0]['centroid']
            #to take old centroid, you need to find time when it happened, and take it from that frame
            current_time = vid.get(1)
            old_time = histories[found_index][0]['time'] 
            vid.set(1, old_time) #set old time   
            flag, frame = vid.read() #read that frame value

            x = old_centroid_x - 14
            y = old_centroid_y - 14
            w = 28
            h = 28
            
            
            found_number = frame[y:y + h,x: x + w, :]  
            #bring back current time
            vid.set(1, current_time)

            if(found_number.shape[0] == 0 or found_number.shape[1] == 0):
                return (False, -1)
            res = extract_numbers(found_number)

            ####TODO: remove
            res = additional_cleaning(res)
            ### TODO End
            res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY) 
           
            res = res / 255            
            
            prediction = predict(res[np.newaxis,:, :, np.newaxis])

            cv2.imwrite(f'slika_{current_time}_{old_time}_{prediction}.bmp', found_number)

            #sub = sub + prediction
            print(f'Number is: {prediction} and on index {found_index} and found history length {len(histories[found_index])}')
            print(f'Time when it was found is {old_time}')

            
            return (True, prediction)

    return (False, -1)


      



def main(video_number):

    # histories = []
    # intersections = {}

    # add = 0
    # sub = 0

    add = 0
    sub = 0     

    
    global histories

    vid = cv2.VideoCapture(f'video-{video_number}.avi')

    flag, number_frame = vid.read()

    #iscrtaj tacke za liniju
    #for green color
    line_frame = process_line(number_frame, GREEN)
    (x1_g,y1_g,x2_g,y2_g) = hough_lines_return(line_frame)

    


    #for blue color
    line_frame = process_line(number_frame, BLUE)
    (x1_b,y1_b,x2_b,y2_b) = hough_lines_return(line_frame)

    #SUBSTRACTION
    lineSegmentGreen = LineSegment(x1_g, y1_g, x2_g, y2_g)
    #ADDITION
    lineSegmentBlue = LineSegment(x1_b, y1_b, x2_b, y2_b)

    intersectionsGreen = {}
    intersectionsBlue = {}
    
    while vid.isOpened():
        flag, number_frame = vid.read()


        time = vid.get(1)

        if(vid.get(cv2.CAP_PROP_FRAME_COUNT) == time):
            break

        digit_frame = extract_numbers(number_frame)

       
        #1)first detect boxes
        bboxes = find_boxes(digit_frame)

        #2)then for each box check if it is close to other history of boxes by eucledian distance       

        for box in bboxes:
            #find centroid for found box
            (x, y, w, h) = box

            (centroid_x, centroid_y) = (x + int(w / 2), y + int(h / 2))

            #a)if it is, append it to it
            found_index = -1 #to check if we found closest centroid in recent history

            #TODO: check eucledian for each item in history, and find the closest one

            for idx, history_item in enumerate(histories):
                    #take last centroid inserted in history_item                
                    (x,y) = history_item[-1]['centroid']
                    last_time_found = history_item[-1]['time']
                    if(eucledianDistance((centroid_x, centroid_y), (x, y)) < 15):
                        histories[idx].append({'centroid' : (centroid_x, centroid_y), 'time' : time})
                        found_index = idx                     
                        break


            ###### TODO: end



            #b)if not, create new item in history specially for it
            if(found_index == -1):
                history_item = []

                #for new line consideration
                R = random.randint(0,255)
                G = random.randint(0,255)
                B = random.randint(0,255)


                new_centroid = {'centroid' : (centroid_x, centroid_y),  'time' : time, 'R' : R, 'G' : G, 'B' : B}
                history_item.append(new_centroid)
                histories.append(history_item)

                found_index = len(histories) - 1


            
            #find intersection places from history items that still haven't been in dictionary


           
            (isIntersect, prediction) = findAndPredictNumberOnLine(number_frame, box, found_index, lineSegmentGreen, intersectionsGreen, vid )
            if(isIntersect):
                sub = sub + prediction

            (isIntersect, prediction) = findAndPredictNumberOnLine(number_frame, box, found_index, lineSegmentBlue, intersectionsBlue, vid )
            if(isIntersect):
                add = add + prediction


            

        #LATER:
        #3)then check how close is box to each line (sum and sub line) 
        #   a) if it is close, add to specific line
    

        #   
                

        # NOW:
        # EXTRA) for each item in history, draw polyline of movement, meaning connect the centroid dots
        for idx,history_item in enumerate(histories):
            point_start = history_item[0]['centroid']

            R = history_item[0]['R']
            G = history_item[0]['G']
            B = history_item[0]['B']
            for jdx, element in enumerate(history_item[1:]):
                point_end  = element['centroid']

                
                cv2.line(number_frame, point_start , point_end, (R, G, B) )
                point_start = point_end




        for key in intersectionsGreen:
            cv2.circle(number_frame, intersectionsGreen[key], 5, color = (0,255,0), thickness = 3)

        for key in intersectionsBlue:
            cv2.circle(number_frame, intersectionsBlue[key], 5, color = (0,0,255), thickness = 3)

        

        #4) calculate sum and sub and total separately and view in box

        #ADD
        cv2.putText(number_frame, 'add: ' + str(add), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        #SUB
        cv2.putText(number_frame, 'sub: ' + str(sub) , (0, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        #TOTAL
        total = add - sub
        cv2.putText(number_frame, 'total: ' + str(total), (0, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)


        lineSegmentGreen.drawLine(cv2, number_frame)
        lineSegmentBlue.drawLine(cv2, number_frame, (255,0,0))

        for box in bboxes:
            (x,y,w,h) = box

            #chosen_color = (0,255,0) if lineSegmentBlue.getLineCrossed(x,y) else (255,0,0)
            chosen_color = (0,255,0) if True else (255,0,0)
            
            cv2.rectangle(number_frame, (x,y), (x + w, y + h),color = chosen_color)
            
        cv2.imshow('Video', number_frame)


        if cv2.waitKey(1) & 0xFF == 27 or flag==False:
            break

    text_file = open("rezultat_dusan.txt", "a") 
    text_file.write(f'video-{video_number}.avi\t{total}\n')
    text_file.close()

    cv2.destroyAllWindows()
    vid.release()     

    #return total

# text_file = open("rezultat_dusan.txt", "a") 
# text_file.write(f'RA 196/2015 Dusan Svilarkovic\n')
# text_file.write(f'file\tsum\n')
# text_file.close()
#for i in range(0,10):
import sys as sys

main(int(sys.argv[1]))