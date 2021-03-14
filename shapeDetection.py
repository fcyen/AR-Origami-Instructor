import cv2
import json
from matplotlib import pyplot as plt
import numpy as np

import draw

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)

# retrieve saved values
with open('trackbarValues.json') as json_file:
    raw = json.load(json_file)
    hsv_green = raw[str(3)]  # green colour
    lowerHSV_green = np.array(hsv_green["LowerHSV"])
    upperHSV_green = np.array(hsv_green["UpperHSV"])


def detectShape(mask, dimg=[], minArea=1000):
    '''
    Detects contours and draw an outline around it

        Parameters:
            mask: binary mask after applying HSV range
            dimg: image to draw contour on 

        Returns: 
            result: contours larger than minArea (default 1000)      
    '''
    result = []

    # remove noise
    kernel = np.ones((5, 5), np.uint8)
    cv2.erode(mask, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > minArea:  # remove noise
            approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
            result.append(approx)

            if len(dimg) > 0:
                cv2.drawContours(dimg, [approx], 0, draw.DEBUG_GREEN, 2)

    return result, _


# currently unused
def detectLines(img, t1, t2, dimg):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.GaussianBlur(img, (3, 3), 0, imgGray)

    # this is to recognize white on white
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
    #dilated = cv2.dilate(imgGray, kernel)

    edges = cv2.Canny(imgGray, t1, t2, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80,
                            minLineLength=50, maxLineGap=20)

    cv2.imshow('Edges', edges)

    try:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(dimg, (x1, y1), (x2, y2), (0, 255, 0), 2)
    except:
        # when line is a NoneType
        pass


def findSquare(mask, step_num, dimg=[]):
    ''' Finds square in image and draws an outline (debug), returns the reshaped square contour '''
    contours, _ = detectShape(mask, dimg)

    for cnt in contours:
        if len(cnt) == 4:
            shape = cnt.reshape(4, 2)
            a, b, c, d = shape
            l1 = calculatedSquaredDistance(a, b)
            l2 = calculatedSquaredDistance(b, c)
            l3 = calculatedSquaredDistance(c, d)
            l4 = calculatedSquaredDistance(d, a)
            # check if the edges are equal length
            v1 = abs(l1-l3) < (0.3*l1)
            v2 = abs(l2-l4) < (0.3*l2)
            v3 = abs(l2-l1) < (0.3*l2)
            if v1 and v2 and v3:
                return shape
            elif len(dimg) > 0:
                print('l1: {}, l2: {}, l3: {}, l4: {}'.format(l1, l2, l3, l4))
    return []


def findTriangle(mask, dimg=[]):
    ''' Finds triangle in image and draws an outline (debug), returns the reshaped triangle contour '''
    contours, _ = detectShape(mask, dimg)
    for cnt in contours:
        if len(cnt) == 3:
            shape = cnt.reshape(3, 2)
            a, b, c = shape
            l1 = calculatedSquaredDistance(a, b)
            l2 = calculatedSquaredDistance(b, c)
            l3 = calculatedSquaredDistance(c, a)
            # check if a2 + b2 = c2
            v1 = abs(l1 - (l2+l3)) < (0.2*l1)
            v2 = abs(l2 - (l1+l3)) < (0.2*l2)
            v3 = abs(l3 - (l2+l1)) < (0.2*l3)
            if v1 or v2 or v3:
                return shape
            elif len(dimg) > 0:
                print('l1: {}, l2: {}, l3: {}'.format(l1, l2, l3))
    return []


def findTriangleWithFold(mask, dimg=[], debug=False):
    ''' Finds triangle in image and draw an outline (debug), returns the triangle contour, with the top vertex at index 0 '''
    if debug:
        contours, _ = detectShape(mask, dimg)
    else:
        contours, _ = detectShape(mask)

    for cnt in contours:
        if len(cnt) == 5:   # paper is slightly open
            hull = cv2.convexHull(cnt, returnPoints=False)
            if len(hull) == 4:
                x = 10 - hull.sum()  # find out concave vertex index
                a = cnt[(x+1) % 5]  # next to x
                b = cnt[(x+2) % 5]
                c = cnt[(x+3) % 5]
                d = cnt[(x+4) % 5]  # next to x
                l1 = calculatedSquaredDistance(a[0], c[0])
                l2 = calculatedSquaredDistance(b[0], d[0])

                if l1 > l2:  # ac are bases
                    shape = np.array([b, c, a])
                else:       # bd are bases
                    shape = np.array([c, b, d])

                if debug:
                    cv2.drawContours(dimg, [shape], 0, draw.DEBUG_GREEN, 2)
                return shape

            else:
                print('hull points: {}'.format(len(hull)))

        elif len(cnt) == 3:
            a, b, c = cnt
            l1 = calculatedSquaredDistance(a[0], b[0])
            l2 = calculatedSquaredDistance(b[0], c[0])
            l3 = calculatedSquaredDistance(c[0], a[0])

            # check if a2 + b2 = c2
            v1 = abs(l1 - (l2+l3)) < (0.2*l1)
            if v1:  # ab is the long edge
                return np.array([c, a, b])

            v2 = abs(l2 - (l1+l3)) < (0.2*l2)
            if v2:  # bc is the long edge
                return np.array([a, b, c])

            v3 = abs(l3 - (l2+l1)) < (0.2*l3)
            if v3:  # ac is the long edge
                return np.array([b, a, c])

    return []


def identifyTriangle(mask, dimg=[], debug=False, step_num=3):
    ''' Finds triangle in image and draw an outline (debug), returns the triangle contour, with the top vertex at index 0 '''
    if debug:
        contours, hierarchy = detectShape(mask, dimg, minArea=0)
    else:
        contours, hierarchy = detectShape(mask, minArea=0)

    max_area = 0
    max_cnt = []
    max_i = 0

    for i in range(len(contours)):
        # find contour with the largest area
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_i = i
            max_area = area
    max_cnt = contours[i]

    if len(max_cnt) == 5:   # paper is slightly open
        if step_num == 2:
            return []
        elif step_num == 3:
            hull = cv2.convexHull(max_cnt, returnPoints=False)
            defects = cv2.convexityDefects(max_cnt, hull)   # [ start point, end point, farthest point, approximate distance to farthest point ]

            if defects is not None and len(defects) == 1 and defects[0][3] < 100:
                # correct shape
                # >> see distance
                # >> see if start point > end point or vice versa
                print(defects[0][3])

                # assuming end point is larger
                x = defects[0][1]
                a = max_cnt[(x+1) % 5]  # next to x
                b = max_cnt[(x+2) % 5]
                c = max_cnt[(x+3) % 5]
                d = max_cnt[(x+4) % 5]  # next to x
                l1 = calculatedSquaredDistance(a[0], c[0])
                l2 = calculatedSquaredDistance(b[0], d[0])

                if l1 > l2:  # ac are bases
                    shape = np.array([b, c, a])
                else:       # bd are bases
                    shape = np.array([c, b, d])


    elif len(max_cnt) == 3:
        # check if there are two child contours
        # hierarchy = [next, previous, first child, parent]
        child_index = hierarchy[max_i][2]
        if child_index > 0:
            # >> use if necessary
            # num_of_children = 0
            # while child_index > 0:
            #     child_cnt = contours[child_index]
            #     child_area = cv2.contourArea(child_cnt)
            #     if child_area > 100 and child_area < 700: 
            #         num_of_children += 1

            #     child_index = hierarchy[child_index][0] # fetch next child

            if step_num == 2:
                return []
                
        a, b, c = max_cnt
        l1 = calculatedSquaredDistance(a[0], b[0])
        l2 = calculatedSquaredDistance(b[0], c[0])
        l3 = calculatedSquaredDistance(c[0], a[0])

        # check if a2 + b2 = c2
        v1 = abs(l1 - (l2+l3)) < (0.2*l1)
        if v1:  # ab is the long edge
            return np.array([c, a, b])

        v2 = abs(l2 - (l1+l3)) < (0.2*l2)
        if v2:  # bc is the long edge
            return np.array([a, b, c])

        v3 = abs(l3 - (l2+l1)) < (0.2*l3)
        if v3:  # ac is the long edge
            return np.array([b, a, c])

    return []


def calculatedSquaredDistance(pt1, pt2):
    return (pt1[0]-pt2[0]) ** 2 + (pt1[1]-pt2[1]) ** 2


# img = cv2.imread('paper-on-desk.jpg')
# imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# lower_hsv = np.array([0,0,0])
# upper_hsv = np.array([179,42,255])
# mask = cv2.inRange(imgHSV, lower_hsv, upper_hsv)

# detectShape(img, mask, img)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.show()

# while True:
#     keyPressed = cv2.waitKey(1)
#     # quit
#     if (keyPressed & 0xFF) == ord('q'):
#         cv2.destroyAllWindows()
#         break
#     cv2.imshow("Img", img)
#     detectShape2(img, 20, 80)
