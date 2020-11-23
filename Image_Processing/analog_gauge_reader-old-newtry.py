import cv2
import numpy as np

filename = 'images/Gauge6.jpg'

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

def preprocess(gauge_number, file_type):

    img = cv2.imread(filename)
    height, width, channels = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #convert to gray

    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,400,
                            param1=15,param2=40,minRadius=10,maxRadius=0)
    # average found circles, found it to be more accurate than trying to tune HoughCircles parameters to get just the right one
    a, b, c = circles.shape
    x,y,r = avg_circles(circles, b)
    #draw center and circle
    r1 = x-r-20
    r2 = y-r-20
    r3 = x+r+20
    r4 = y+r+20
    cv2.circle(img, (x, y), r, (0, 0, 255), 3, cv2.LINE_AA)  # draw circle
    cv2.circle(img, (x, y), 2, (0, 255, 0), 3, cv2.LINE_AA)  # draw center of circle

    separation = 10.0 #in degrees
    interval = int(360 / separation)
    p1 = np.zeros((interval,2))  #set empty arrays
    p2 = np.zeros((interval,2))
    p_text = np.zeros((interval,2))
    for i in range(0,interval):
        for j in range(0,2):
            if (j%2==0):
                p1[i][j] = x + 0.9 * r * np.cos(separation * i * 3.14 / 180) #point for lines
            else:
                p1[i][j] = y + 0.9 * r * np.sin(separation * i * 3.14 / 180)
    text_offset_x = 10
    text_offset_y = 5
    for i in range(0, interval):
        for j in range(0, 2):
            if (j % 2 == 0):
                p2[i][j] = x + r * np.cos(separation * i * 3.14 / 180)
                p_text[i][j] = x - text_offset_x + 1.2 * r * np.cos((separation) * (i+9) * 3.14 / 180) #point for text labels, i+9 rotates the labels by 90 degrees
            else:
                p2[i][j] = y + r * np.sin(separation * i * 3.14 / 180)
                p_text[i][j] = y + text_offset_y + 1.2* r * np.sin((separation) * (i+9) * 3.14 / 180)  # point for text labels, i+9 rotates the labels by 90 degrees

    #add the lines and labels to the image
    for i in range(0,interval):
        cv2.line(img, (int(p1[i][0]), int(p1[i][1])), (int(p2[i][0]), int(p2[i][1])),(0, 255, 0), 2)
        cv2.putText(img, '%s' %(int(i*separation)), (int(p_text[i][0]), int(p_text[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,0,0),1,cv2.LINE_AA)


    #cv2.imwrite('gauge-%s-calibration.%s' % (gauge_number, file_type), img)

    return  x, y, r

def calibrate_gauge(gauge_number, file_type):
    min_angle = 60
    max_angle = 300
    min_value = 0
    max_value = 240
    units = 'psi'
    return min_angle, max_angle, min_value, max_value, units,

def get_current_value(img, min_angle, max_angle, min_value, max_value, x, y, r, gauge_number, file_type):

    #for testing purposes
    img = cv2.imread(filename)

    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Set threshold and maxValue
    thresh = 175
    maxValue = 255

    th, dst2 = cv2.threshold(gray2, thresh, maxValue, cv2.THRESH_BINARY_INV);

    cv2.imwrite('dst.jpg', dst2)

    # find lines
    minLineLength = 9
    maxLineGap = 0
    lines = cv2.HoughLinesP(image=dst2, rho=3, theta=np.pi / 180, threshold=500,minLineLength=minLineLength, maxLineGap=0)  # rho is set to 3 to detect more lines, easier to get more then filter them out later
    print(len(lines))

    final_line_list = []
    #print "radius: %s" %r

    diff1LowerBound = 0 #diff1LowerBound and diff1UpperBound determine how close the line should be from the center
    diff1UpperBound = 0.3
    diff2LowerBound = 0.1 #diff2LowerBound and diff2UpperBound determine how close the other point of the line should be to the outside of the gauge
    diff2UpperBound = 2.0

    for i in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            diff1 = dist_2_pts(x, y, x1, y1)  # x, y is center of circle
            diff2 = dist_2_pts(x, y, x2, y2)  # x, y is center of circle
            #set diff1 to be the smaller (closest to the center) of the two), makes the math easier
            if (diff1 > diff2):
                temp = diff1
                diff1 = diff2
                diff2 = temp
            # check if line is within an acceptable range
            if (((diff1<diff1UpperBound*r) and (diff1>diff1LowerBound*r) and (diff2<diff2UpperBound*r)) and (diff2>diff2LowerBound*r)):
                line_length = dist_2_pts(x1, y1, x2, y2)
                # add to final list
                final_line_list.append([x1, y1, x2, y2])

    x1 = final_line_list[0][0]
    y1 = final_line_list[0][1]
    x2 = final_line_list[0][2]
    y2 = final_line_list[0][3]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite('gauge-%s-lines-2.%s' % (gauge_number, file_type), img)

    #find the farthest point from the center to be what is used to determine the angle
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
    #np.rad2deg(res) #coverts to degrees

    res = np.rad2deg(res)
    if x_angle > 0 and y_angle > 0:  #in quadrant I
        final_angle = 270 - res
    if x_angle < 0 and y_angle > 0:  #in quadrant II
        final_angle = 90 - res
    if x_angle < 0 and y_angle < 0:  #in quadrant III
        final_angle = 90 - res
    if x_angle > 0 and y_angle < 0:  #in quadrant IV
        final_angle = 270 - res

    #print final_angle

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

    return new_value,level

def main():
    gauge_number = 2
    file_type='jpg'
    x, y, r = preprocess(gauge_number, file_type)
    min_angle, max_angle, min_value, max_value, units = calibrate_gauge(gauge_number, file_type)

    #feed an image (or frame) to get the current value, based on the calibration, by default uses same image as calibration
    img = cv2.imread(filename)
    val,level = get_current_value(img, min_angle, max_angle, min_value, max_value, x, y, r, gauge_number, file_type)
    print ("Current reading: %s %s" %(val, units))
    print ("Gauge Reading Level : %s" %(level))

if __name__=='__main__':
    main()