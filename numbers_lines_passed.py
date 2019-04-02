import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
import math
import random
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

    
    (x1,y1,x2,y2) = lines[0,0,:]
    
    return (x1,y1,x2,y2)

BLUE = 0
GREEN = 1
EPSILON_Y = 5
EPSILON_D = 15

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
        # if(x in x_range): #and y in y_range):
        #     return True

        return False

    def drawLine(self, cv2, frame):
        cv2.line(frame, self.point_begin, self.point_end, color = (0, 255, 0), thickness = 3)
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


def findAndPredictNumberOnLine(frame, bbox, found_index, lineSegment, intersections):
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
    if(d < EPSILON_D and (found_index not in intersections.keys())):
        #keep intersection place
        #print(Intersection found (wrong one)!')
        if(lineSegment.pointIsClose(centroid) == True):


            intersections[found_index] = centroid
            print('Intersection REALLY found!')

            (old_centroid_x, old_centroid_y) = histories[found_index][-1]['centroid']
            x = old_centroid_x - 14
            y = old_centroid_y - 14
            w = 28
            h = 28

            found_number = frame[y:y + h,x: x + w, :]
            res = extract_numbers(found_number)
            res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY) 
            res = res / 255

            
            
            prediction = predict(res[np.newaxis,:, :, np.newaxis])

            cv2.imwrite('slika' + str(prediction) +'_' + str(found_index) + '.bmp', found_number)
            #sub = sub + prediction
            print(f'Number is: {prediction} and on index {found_index} and found history length {len(histories[found_index])}')

            
            return (True, prediction)

    return (False, -1)


      



def main():

    # histories = []
    # intersections = {}

    # add = 0
    # sub = 0

    add = 0
    sub = 0     

    
    global histories

    vid = cv2.VideoCapture('video-0.avi')

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

    #ini # sub = 0
    # add = 0tialize array of histories
    #each item is array of pairs of centroid and time (frame_index) when they were found
    # histories = []
    
    # intersections = {}
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

            #check eucledian for each item in history
            for idx, history_item in enumerate(histories):
                #take last centroid inserted in history_item
                (x,y) = history_item[-1]['centroid']
                if(eucledianDistance((centroid_x, centroid_y), (x, y)) < 15):
                    histories[idx].append({'centroid' : (centroid_x, centroid_y), 'time' : time})
                    found_index = idx
                    
                    break

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

            
            (isIntersect, prediction) = findAndPredictNumberOnLine(number_frame, box, found_index, lineSegmentGreen, intersectionsGreen )
            if(isIntersect):
                sub = sub + prediction

            (isIntersect, prediction) = findAndPredictNumberOnLine(number_frame, box, found_index, lineSegmentBlue, intersectionsBlue )
            if(isIntersect):
                add = add + prediction
            #If history is not so big
            # if(len(histories[found_index]) <= 10):
            #     continue
            # centroid = (centroid_x, centroid_y)
            # d = lineSegmentGreen.getShortestDistance(centroid)
            # if(d < EPSILON_D and (found_index not in intersections.keys()) and lineSegmentGreen.pointIsClose(centroid) == True):
              
            #     intersections[found_index] = centroid
            #     print('Intersection REALLY found!')

            #     (old_centroid_x, old_centroid_y) = histories[found_index][-1]['centroid']
            #     x = old_centroid_x - 14
            #     y = old_centroid_y - 14
            #     w = 28
            #     h = 28

            #     found_number = number_frame[y:y + h,x: x + w, :]
            #     res = extract_numbers(found_number)
            #     res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY) 
            #     res = res / 255

            #     cv2.imwrite('slika' + str(found_index) + '.bmp', found_number)
                
            #     prediction = predict(res[np.newaxis,:, :, np.newaxis])
            #     sub = sub + prediction
            #     print(f'Number is: {prediction} and on index {found_index} and found history length {len(histories[found_index])}')

   


            

        #LATER:
        #3)then check how close is box to each line (sum and sub line) 
        #   a) if it is close, add to specific line
    

        #   
                

        # NOW:
        # EXTRA) for each item in history, draw polyline of movement, meaning connect the centroid dots
        for history_item in histories:
            point_start = history_item[0]['centroid']

            R = history_item[0]['R']
            G = history_item[0]['G']
            B = history_item[0]['B']
            for element in history_item[1:]:
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
        

        for box in bboxes:
            (x,y,w,h) = box

            #chosen_color = (0,255,0) if lineSegmentBlue.getLineCrossed(x,y) else (255,0,0)
            chosen_color = (0,255,0) if True else (255,0,0)
            
            cv2.rectangle(number_frame, (x,y), (x + w, y + h),color = chosen_color)
            
        cv2.imshow('video-0.avi', number_frame)


        if cv2.waitKey(1) & 0xFF == 27 or flag==False:
            break

    text_file = open("rezultat_dusan.txt", "w") 
    text_file.write("Total: %d" % total)
    text_file.close()

    cv2.destroyAllWindows()
    vid.release()     



main()