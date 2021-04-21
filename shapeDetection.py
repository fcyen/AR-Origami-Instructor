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


def detectShape(mask, dimg=[], minArea=1000, debug=False):
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

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
            result = approx

    return result


def detectContours(mask, dimg=[]):
    '''
    Detects contours and draw an outline around it

        Parameters:
            mask: binary mask after applying HSV range
            dimg: image to draw contour on 

        Returns: 
            contours, hierarchy    
    '''
    result = []

    # remove noise
    kernel = np.ones((5, 5), np.uint8)
    cv2.erode(mask, kernel)

    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours, hierarchy


def getLargestShape(shapes):
    max_area = 0
    max_shape = []
    for shape in shapes:
        area = cv2.contourArea(shape)
        if area > max_area:
            max_area = area
            max_shape = shape
    return max_shape


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


def findSquare(mask, step_num, dimg=[], debug=False):
    ''' Finds square in image and draws an outline (debug), returns the reshaped square contour '''
    contours = detectShape(mask, dimg, debug)

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
            elif debug:
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


def identifyTriangle(mask, step_num, dimg=[], debug=False):
    ''' Finds triangle in image and draw an outline (debug), returns the triangle contour, with the top vertex at index 0 '''
    if debug:
        contours, hierarchy = detectContours(mask, dimg)
    else:
        contours, hierarchy = detectContours(mask)

    max_area = 0
    max_cnt_full = []
    max_i = -1

    for i in range(len(contours)):
        # find contour with the largest area
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_i = i
            max_area = area

    if max_i > 0:
        max_cnt_full = contours[max_i]
        max_cnt = cv2.approxPolyDP(
            max_cnt_full, 0.02*cv2.arcLength(max_cnt_full, True), True)

        if len(max_cnt) == 5:   # paper is slightly open
            if step_num == 1:
                return []
            elif step_num == 0:
                hull = cv2.convexHull(max_cnt, returnPoints=False)
                # [ start point, end point, farthest point, approximate distance to farthest point ]
                defects = cv2.convexityDefects(max_cnt, hull)

                if defects is not None and len(defects) == 1 and defects[0][0][3] < 100:
                    # correct shape
                    # >> see distance
                    # >> see if start point > end point or vice versa
                    print(defects[0][0][3])

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
            with open('trackbarValues.json') as json_file:
                raw = json.load(json_file)
                hsv = raw[str(3)]
                lowerHSV = np.array(hsv["LowerHSV"])
                upperHSV = np.array(hsv["UpperHSV"])
            img_hsv = cv2.cvtColor(dimg, cv2.COLOR_BGR2HSV)
            img_masked = cv2.inRange(img_hsv, lowerHSV, upperHSV)

            small_cnt = detectShape(img_masked, dimg, minArea=300)
            marker_count = 0
            for cnt in small_cnt:
                area = cv2.contourArea(cnt)
                if area > 300 and area < 700:
                    marker_count += 1
                    # cv2.drawContours(dimg, [cnt], 0, (0,0,255))

            # child_index = hierarchy[0][max_i][2]
            # if child_index > 0:
            #     # >> use if necessary
            #     num_of_children = 0
            #     while child_index > 0:
            #         child_cnt = contours[child_index]
            #         child_area = cv2.contourArea(child_cnt)
            #         if child_area > 200 and child_area < 500:
            #             num_of_children += 1
            #             cv2.drawContours(dimg, [child_cnt], 0, (0,0,150))
            #             cv2.putText(dimg, str(child_area), (child_cnt[0][0][0], child_cnt[0][0][1]), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,150))

            #         child_index = hierarchy[0][child_index][0] # fetch next child

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


def differentiateTriangle(mask, step_num, dimg, debug=False):
    ''' Differentiate between step 2 and step 3's triangle '''
    contours = detectShape(mask, dimg, minArea=5000, debug=True)
    largest_cnt = getLargestShape(contours)

    if len(largest_cnt) == 3 or len(largest_cnt) == 5:
        # check if markers is shown
        with open('trackbarValues.json') as json_file:
            raw = json.load(json_file)
            hsv = raw[str(3)]
            lowerHSV = np.array(hsv["LowerHSV"])
            upperHSV = np.array(hsv["UpperHSV"])

        img_hsv = cv2.cvtColor(dimg, cv2.COLOR_BGR2HSV)
        img_marker = cv2.inRange(img_hsv, lowerHSV, upperHSV)
        markers, _ = detectContours(img_marker, dimg)
        marker_count = 0
        for m in markers:
            area = cv2.contourArea(m)
            if area > 200 and area < 900:
                marker_count += 1

        if marker_count < 2 and step_num == 2:
            return True
        elif marker_count == 2 and step_num == 3:
            return True

    return False


def calculatedSquaredDistance(pt1, pt2):
    return (pt1[0]-pt2[0]) ** 2 + (pt1[1]-pt2[1]) ** 2
