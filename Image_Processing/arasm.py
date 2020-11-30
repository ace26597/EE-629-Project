
'''
from __future__ import print_function

import csv
import time
import imutils
import schedule
import pytesseract
import numpy as np
import cv2

white = [255, 255, 255]

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

tessdata_dir_config = '--tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"'

lower_range = np.array([0,100,100], dtype=np.uint8)
upper_range = np.array([10, 255, 255], dtype=np.uint8)
data=list()
lower_red = np.array([170,100,100])
upper_red = np.array([180,255,255])
min_angle = 60
max_angle = 300
min_value = 0
max_value = 240
units = 'psi'
# load the image, convert it to grayscale, and blur it
filename = 'Image_Processing/images/gauge4.jpg'
image = cv2.imread(filename)

def avg_circles(circles, b):
    avg_x=0
    avg_y=0
    avg_r=0
    for i in range(b):
        #optional - average for multiple circles (can happen when a gauge is at a slight angle)
        avg_x = avg_x + circles[0][i][0]
        avg_y = avg_y + circles[0][i][1]
        avg_r = avg_r + circles[0][i][2]
    avg_x = int(avg_x/(b))
    avg_y = int(avg_y/(b))
    avg_r = int(avg_r/(b))
    return avg_x, avg_y, avg_r

def dist_2_pts(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_current_value(gray, min_angle, max_angle, min_value, max_value, x, y, r):
    # apply thresholding which helps for finding lines
    image = cv2.imread(filename)
    gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    th, dst2 = cv2.threshold(gray2, 175, 255, cv2.THRESH_BINARY_INV);
    cv2.imwrite('dst1.jpg', dst2)
    minLineLength = 8
    lines = cv2.HoughLinesP(image=dst2, rho=3, theta=np.pi / 180, threshold=100, minLineLength=minLineLength,maxLineGap=0)

    if lines is not None:
        final_line_list = []
        diff1LowerBound = 0  # diff1LowerBound and diff1UpperBound determine how close the line should be from the center
        diff1UpperBound = 0.6
        diff2LowerBound = 0.6  # diff2LowerBound and diff2UpperBound determine how close the other point of the line should be to the outside of the gauge
        diff2UpperBound = 4.0

        for i in range(0, len(lines)):
            for x1, y1, x2, y2 in lines[i]:
                diff1 = dist_2_pts(x, y, x1, y1)  # x, y is center of circle
                diff2 = dist_2_pts(x, y, x2, y2)  # x, y is center of circle
                # set diff1 to be the smaller (closest to the center) of the two), makes the math easier
                if (diff1 > diff2):
                    temp = diff1
                    diff1 = diff2
                    diff2 = temp
                # check if line is within an acceptable range
                if (((diff1 < diff1UpperBound * r) and (diff1 > diff1LowerBound * r) and (
                        diff2 < diff2UpperBound * r)) and (diff2 > diff2LowerBound * r)):
                    final_line_list.append([x1, y1, x2, y2])

        if len(final_line_list)<2:
            return "error","error"
        x1 = final_line_list[0][0]
        y1 = final_line_list[0][1]
        x2 = final_line_list[0][2]
        y2 = final_line_list[0][3]
        cv2.line(dst2, (x1, y1), (x2, y2), (0, 255, 255), 2)
        #cv2.imshow('gauge', image)
        # find the farthest point from the center to be what is used to determine the angle
        dist_pt_0 = dist_2_pts(x, y, x1, y1)
        dist_pt_1 = dist_2_pts(x, y, x2, y2)
        if (dist_pt_0 > dist_pt_1):
            x_angle = x1 - x
            y_angle = y - y1
        else:
            x_angle = x2 - x
            y_angle = y - y2
        # take the arc tan of y/x to find the angle
        res = np.arctan(np.divide(float(y_angle), float(x_angle)))
        #deg = np.rad2deg(res)

        if x_angle > 0 and y_angle > 0:  # in quadrant I
            final_angle = 270 - res
        elif x_angle < 0 and y_angle > 0:  # in quadrant II
            final_angle = 90 - res
        elif x_angle < 0 and y_angle < 0:  # in quadrant III
            final_angle = 90 - res
        elif x_angle > 0 and y_angle < 0:  # in quadrant IV
            final_angle = 270 - res
        else:
            return "error", "error"
        # print final_angle
        old_min = float(min_angle)
        old_max = float(max_angle)
        new_min = float(min_value)
        new_max = float(max_value)

        old_value = final_angle

        old_range = (old_max - old_min)
        new_range = (new_max - new_min)
        new_value = (((old_value - old_min) * new_range) / old_range) + new_min
        if new_value < new_max * 0.33:
            level = 'LOW'
        elif new_value < new_max * 0.66:
            level = 'MEDIUM'
        else:
            level = 'HIGH'
    else:
        print('wtf')
    return new_value, level

def detectShape(c):
    shape = 'unknown'
    # calculate perimeter using
    peri = cv2.arcLength(c, True)
    # apply contour approximation and store the result in vertices
    vertices = cv2.approxPolyDP(c, 0.04 * peri, True)

    # If the shape it triangle, it will have 3 vertices
    if len(vertices) == 3:
        shape = 'triangle'

    # if the shape has 4 vertices, it is either a square or
    # a rectangle
    elif len(vertices) == 4:
        # using the boundingRect method calculate the width and height
        # of enclosing rectange and then calculte aspect ratio

        x, y, width, height = cv2.boundingRect(vertices)
        aspectRatio = float(width) / height

        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        if aspectRatio >= 0.95 and aspectRatio <= 1.05:
            shape = "square"
        else:
            shape = "rectangle"

    # if the shape is a pentagon, it will have 5 vertices
    elif len(vertices) == 5:
        shape = "pentagon"

    # otherwise, we assume the shape is a circle
    else:
        shape = "circle"

    # return the name of the shape
    return shape

text = pytesseract.image_to_string(image, lang='eng')
print(text)

grayScale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

barlevel=list()
gaugeval=list()
gaugelevel=list()
circles = cv2.HoughCircles(grayScale, cv2.HOUGH_GRADIENT, 1.2, 1000,param1=15, param2=40, minRadius=10, maxRadius=0)

circle = np.round(circles[0,:]).astype("int")

# loop over the (x, y) coordinates and radius of the circles
for (x, y, r) in circle:
    cv2.circle(image, (x, y), r, (0, 255, 0), 4)
    val, level = get_current_value(image, min_angle, max_angle, min_value, max_value, x, y, r)
    print("Current reading: %s %s" % (int(val), units))
    print("Gauge Reading Level : %s" % (level))
    gaugelevel.append(level)
    gaugeval.append(str(int(val)))

cv2.imwrite('circle.jpg', image)

sigma = 0.33
v = np.median(grayScale)
low = int(max(0, (1.0 - sigma) * v))
high = int(min(255, (1.0 + sigma) * v))
edged = cv2.Canny(grayScale, low, high)
(_, cnts, _) = cv2.findContours(edged,
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# loop over the contours
for c in cnts:
    # compute the moment of contour
    M = cv2.moments(c)
    shape = detectShape(c)

    if(shape == "rectangle"):
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(c)  # offsets - with this you get 'mask'
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #cv2.imshow('cutted contour', image[y:y + h, x:x + w])
        crop_img = image[y:y + h, x:x + w]
        # show the output image
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        mask0 = cv2.inRange(hsv, lower_range, upper_range)
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        size = area-perimeter
        mask = mask0 + mask1
        PixelsInRange = cv2.countNonZero(mask)
        div = PixelsInRange + perimeter
        frac_red = np.divide(float(div), int(size))
        percent_red = np.multiply((float(frac_red)), 100)

        percent_red = int(percent_red)
        barlevel.append(str(percent_red))
        print('Bar Indicator Level : ' + str(percent_red) + ' %')
        #cv2.imwrite('mask.png',mask)

i = 0

text = text.splitlines()
for h in zip(text):
    data.append(h)
for i in zip(gaugeval):
    data.append(i)
for j in zip(gaugelevel):
    data.append(j)
for k in zip(barlevel):
    data.append(k)

finaldata = []
data = [val for sublist in data for val in sublist]
print(data)

with open('arasm.csv', 'a') as csvFile:
    writer = csv.writer(csvFile,delimiter = ',')
    writer.writerow(data)

cv2.waitKey(0)


'''

from __future__ import print_function

import csv
import time
import imutils
import schedule
import pytesseract
import numpy as np
import cv2

white = [255, 255, 255]

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

tessdata_dir_config = '--tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"'

lower_range = np.array([0,100,100], dtype=np.uint8)
upper_range = np.array([10, 255, 255], dtype=np.uint8)

lower_red = np.array([170,100,100])
upper_red = np.array([180,255,255])
min_angle = 60
max_angle = 300
min_value = 0
max_value = 240
units = 'psi'
# load the image, convert it to grayscale, and blur it
filename = 'Image_Processing/images/5.png'
image = cv2.imread(filename)

def avg_circles(circles, b):
    avg_x=0
    avg_y=0
    avg_r=0
    for i in range(b):
        #optional - average for multiple circles (can happen when a gauge is at a slight angle)
        avg_x = avg_x + circles[0][i][0]
        avg_y = avg_y + circles[0][i][1]
        avg_r = avg_r + circles[0][i][2]
    avg_x = int(avg_x/(b))
    avg_y = int(avg_y/(b))
    avg_r = int(avg_r/(b))
    return avg_x, avg_y, avg_r

def dist_2_pts(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_current_value(gray, min_angle, max_angle, min_value, max_value, x, y, r):
    # apply thresholding which helps for finding lines
    image = cv2.imread(filename)
    gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    th, dst2 = cv2.threshold(gray2, 175, 255, cv2.THRESH_BINARY_INV);
    cv2.imwrite('dst1.jpg', dst2)
    minLineLength = 8
    lines = cv2.HoughLinesP(image=dst2, rho=3, theta=np.pi / 180, threshold=50, minLineLength=minLineLength,maxLineGap=0)

    if lines is not None:
        final_line_list = []
        diff1LowerBound = 0  # diff1LowerBound and diff1UpperBound determine how close the line should be from the center
        diff1UpperBound = 0.6
        diff2LowerBound = 0.1  # diff2LowerBound and diff2UpperBound determine how close the other point of the line should be to the outside of the gauge
        diff2UpperBound = 4.0

        for i in range(0, len(lines)):
            for x1, y1, x2, y2 in lines[i]:
                diff1 = dist_2_pts(x, y, x1, y1)  # x, y is center of circle
                diff2 = dist_2_pts(x, y, x2, y2)  # x, y is center of circle
                # set diff1 to be the smaller (closest to the center) of the two), makes the math easier
                if (diff1 > diff2):
                    temp = diff1
                    diff1 = diff2
                    diff2 = temp
                # check if line is within an acceptable range
                if (((diff1 < diff1UpperBound * r) and (diff1 > diff1LowerBound * r) and (
                        diff2 < diff2UpperBound * r)) and (diff2 > diff2LowerBound * r)):
                    final_line_list.append([x1, y1, x2, y2])

        if len(final_line_list)<2:
            return "error","error"
        x1 = final_line_list[0][0]
        y1 = final_line_list[0][1]
        x2 = final_line_list[0][2]
        y2 = final_line_list[0][3]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.imwrite('lines.jpg',image)
        #cv2.imshow('gauge', image)
        # find the farthest point from the center to be what is used to determine the angle
        dist_pt_0 = dist_2_pts(x, y, x1, y1)
        dist_pt_1 = dist_2_pts(x, y, x2, y2)
        if (dist_pt_0 > dist_pt_1):
            x_angle = x1 - x
            y_angle = y - y1
        else:
            x_angle = x2 - x
            y_angle = y - y2
        # take the arc tan of y/x to find the angle
        res = np.arctan(np.divide(float(y_angle), float(x_angle)))
        #deg = np.rad2deg(res)

        if x_angle > 0 and y_angle > 0:  # in quadrant I
            final_angle = 270 - res
        elif x_angle < 0 and y_angle > 0:  # in quadrant II
            final_angle = 90 - res
        elif x_angle < 0 and y_angle < 0:  # in quadrant III
            final_angle = 90 - res
        elif x_angle > 0 and y_angle < 0:  # in quadrant IV
            final_angle = 270 - res
        else:
            return "error", "error"
        # print final_angle
        old_min = float(min_angle)
        old_max = float(max_angle)
        new_min = float(min_value)
        new_max = float(max_value)

        old_value = final_angle

        old_range = (old_max - old_min)
        new_range = (new_max - new_min)
        new_value = (((old_value - old_min) * new_range) / old_range) + new_min
        if new_value < new_max * 0.33:
            level = 'LOW'
        elif new_value < new_max * 0.66:
            level = 'MEDIUM'
        else:
            level = 'HIGH'
    else:
        print('wtf')
    return new_value, level

def detectShape(c):
    shape = 'unknown'
    # calculate perimeter using
    peri = cv2.arcLength(c, True)
    # apply contour approximation and store the result in vertices
    vertices = cv2.approxPolyDP(c, 0.04 * peri, True)

    # If the shape it triangle, it will have 3 vertices
    if len(vertices) == 3:
        shape = 'triangle'

    # if the shape has 4 vertices, it is either a square or
    # a rectangle
    elif len(vertices) == 4:
        # using the boundingRect method calculate the width and height
        # of enclosing rectange and then calculte aspect ratio

        x, y, width, height = cv2.boundingRect(vertices)
        aspectRatio = float(width) / height

        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        if aspectRatio >= 0.95 and aspectRatio <= 1.05:
            shape = "square"
        else:
            shape = "rectangle"

    # if the shape is a pentagon, it will have 5 vertices
    elif len(vertices) == 5:
        shape = "pentagon"

    # otherwise, we assume the shape is a circle
    else:
        shape = "circle"

    # return the name of the shape
    return shape

text = pytesseract.image_to_string(image, lang='eng')
print(text)

grayScale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
th, dst2 = cv2.threshold(grayScale, 175, 255, cv2.THRESH_BINARY_INV);

barlevel=list()

circles = cv2.HoughCircles(dst2, cv2.HOUGH_GRADIENT, 1.2, 1000,param1=15, param2=40, minRadius=12, maxRadius=0)
circle = np.round(circles[0,:]).astype("int")

# loop over the (x, y) coordinates and radius of the circles
for (x, y, r) in circle:
    cv2.circle(image, (x, y), r, (0, 255, 0), 4)
    val, level = get_current_value(image, min_angle, max_angle, min_value, max_value, x, y, r)
    print("Current reading: %s %s" % (int(val), units))
    print("Gauge Reading Level : %s" % (level))
cv2.imwrite('circle.jpg', image)

sigma = 0.33
v = np.median(grayScale)
low = int(max(0, (1.0 - sigma) * v))
high = int(min(255, (1.0 + sigma) * v))
edged = cv2.Canny(grayScale, low, high)
(_, cnts, _) = cv2.findContours(edged,
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# loop over the contours
for c in cnts:
    # compute the moment of contour
    M = cv2.moments(c)
    shape = detectShape(c)

    if(shape == "rectangle"):
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(c)  # offsets - with this you get 'mask'
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #cv2.imshow('cutted contour', image[y:y + h, x:x + w])
        crop_img = image[y:y + h, x:x + w]
        # show the output image
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        mask0 = cv2.inRange(hsv, lower_range, upper_range)
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        size = area-perimeter
        mask = mask0 + mask1
        PixelsInRange = cv2.countNonZero(mask)
        div = PixelsInRange + perimeter
        frac_red = np.divide(float(div), int(size))
        percent_red = np.multiply((float(frac_red)), 100)
        percent_red = int(percent_red)
        barlevel.append(percent_red)
        print('Bar Indicator Level : ' + str(percent_red) + ' %')
        #cv2.imshow('mask',mask)

data = [text,int(val),level,barlevel]
with open('arasm.csv', 'a') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(data)
    csvFile.close()

cv2.waitKey(0)

