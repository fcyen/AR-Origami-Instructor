import cv2
import numpy as np
from matplotlib import pyplot as plt
import styles

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)


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
                cv2.drawContours(dimg, [approx], 0, BLUE, 2)
                #cv2.putText(dimg, str(len(approx)), tuple(cnt[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, BLUE)

    return result


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
    ''' Finds square in image and draw an outline (disabled), returns the reshaped square contour '''
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
            else:
                # cv2. drawContours(dimg, [cnt], 0, GREEN, 3)
                # cv2.imshow('Contour', dimg)
                print('l1: {}, l2: {}, l3: {}, l4: {}'.format(l1, l2, l3, l4))
    return []


def findTriangle(mask, dimg=[]):
    ''' Finds triangle in image and draw an outline (disabled), returns the reshaped triangle contour '''
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
                # cv2.drawContours(dimg, [cnt], 0, GREEN, 3)
                return shape
            else:
                print('l1: {}, l2: {}, l3: {}'.format(l1, l2, l3))
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
