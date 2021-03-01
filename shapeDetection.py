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

    contours, _ = cv2.findContours(
        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > minArea:  # remove noise
            approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
            result.append(approx)

            if len(dimg) > 0:
                cv2.drawContours(dimg, [approx], 0, draw.DEBUG_GREEN, 2)
                #cv2.putText(dimg, str(len(approx)), tuple(cnt[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, draw.DEBUG_GREEN)

    return result

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


def findSquare(mask, dimg=[]):
    ''' Finds square in image and draws an outline (debug), returns the reshaped square contour '''
    contours = detectShape(mask, dimg)

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
    contours = detectShape(mask, dimg)
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
    # img_green = cv2.inRange(dimg, lowerHSV_green, upperHSV_green)

    if debug:
        contours = detectShape(mask, dimg)
        # contours_green = detectShape(img_green, dimg)
    else:
        contours = detectShape(mask)
        # contours_green = detectShape(img_green)

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

        # elif len(cnt) == 3:
        #     correct = False
        #     for cnt_g in contours_green:
        #         if len(cnt_g) == 3:
        #             correct = True
        #             break
        #     if correct:
        #         a, b, c = cnt
        #         l1 = calculatedSquaredDistance(a[0], b[0])
        #         l2 = calculatedSquaredDistance(b[0], c[0])
        #         l3 = calculatedSquaredDistance(c[0], a[0])

        #         # check if a2 + b2 = c2
        #         v1 = abs(l1 - (l2+l3)) < (0.2*l1)
        #         if v1:  # ab is the long edge
        #             return np.array([c, a, b])

        #         v2 = abs(l2 - (l1+l3)) < (0.2*l2)
        #         if v2:  # bc is the long edge
        #             return np.array([a, b, c])

        #         v3 = abs(l3 - (l2+l1)) < (0.2*l3)
        #         if v3:  # ac is the long edge
        #             return np.array([b, a, c])

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
