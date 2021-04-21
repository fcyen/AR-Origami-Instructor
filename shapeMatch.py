import cv2
import math
import numpy as np

import draw

green_contour_size = 10000

def detectContours(bimg, dimg, minArea=1000, approx_val=0.01, debug=False):
    ''' Returns all approximated contours larger than minArea '''
    contours, _ = cv2.findContours(bimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []

    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)

        if area > minArea:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, approx_val*peri, True)
            results.append(approx)
            if debug:
                cv2.drawContours(dimg, [approx], 0, draw.DEBUG_GREEN, draw.THICKNESS_S)

    return results


def getLargestContour(contours):
    max_area = 0
    max_cnt = []

    if len(contours) > 0:
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                max_cnt = cnt
    return max_cnt


def loadReferenceContour(path):
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    ret, img_bin = cv2.threshold(img_blur, 190, 255, cv2.THRESH_BINARY)
    contours = detectContours(img_bin, [], minArea=100, debug=False)
    cnt = getLargestContour(contours)
    return cnt


def compareShapes(cnt1, cnt2, threshold=0.9, debug=False):
    ''' Returns True if both contours are similar '''
    d = -1
    if len(cnt1) > 0 and len(cnt2) > 0:
        d = cv2.matchShapes(cnt1, cnt2, cv2.CONTOURS_MATCH_I2, 0)

    if debug:
        print('d = ' + str(d))

    if d < threshold and d >= 0:
        return True
    else:
        return False


def calculatedSquaredDistance(pt1, pt2):
    return (pt1[0]-pt2[0]) ** 2 + (pt1[1]-pt2[1]) ** 2


def getAccentContour(img, accent_masked):
    contours = detectContours(accent_masked, img)
    cnt = getLargestContour(contours)
    return cnt


def differentiate2and3(img, accent_masked):
    cnt = getAccentContour(img, accent_masked)
    if len(cnt) > 0:
        area = cv2.contourArea(cnt)
        if area < green_contour_size and area > 1000:
            return 3
        else:
            return 2
    else:
        return 2


def getStep3Triangle(img, accent_masked, shape, top, bases):
    accent = getAccentContour(img, accent_masked)
    M = cv2.moments(accent)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    centroid = (cX, cY)

    # find closest vertex
    closest = shape[0] # base2
    min_dist = 1000000
    for point in shape:
        dist = calculatedSquaredDistance(point, centroid)
        if dist < min_dist:
            min_dist = dist
            closest = point

    if closest[0] != bases[0][0]:
        return np.array([[top], [bases[0]], [closest]])
    else:
        return np.array([[top], [bases[1], [closest]]])


    

########################################################################

refcnt_4 = loadReferenceContour('assets/step4a.png')

def identifyCurrentStep(img, img_masked, accent_masked, debug=False):
    ''' Returns:
            - step: [int] current step
                - -1 if no step matches
                - -2 if undecided
            - cnt: [array] original contour         
    '''
    contours = detectContours(img_masked, img, debug)
    cnt = getLargestContour(contours)
    l = len(cnt)

    # =================================================
    if l == 4:
        if debug:
            print('4 vertices')
        shape = cnt.reshape(4, 2)
        a, b, c, d = shape

        # check if the edges are equal length
        l1 = calculatedSquaredDistance(a, b)
        l2 = calculatedSquaredDistance(b, c)
        l3 = calculatedSquaredDistance(c, d)
        l4 = calculatedSquaredDistance(d, a)
        v1 = abs(l1-l3) < (0.3*l1)
        v2 = abs(l2-l4) < (0.3*l2)
        v3 = abs(l2-l1) < (0.3*l2)
        if v1 and v2 and v3:
            return 1, cnt
        elif debug:
            print('l1: {}, l2: {}, l3: {}, l4: {}'.format(l1, l2, l3, l4))

    # ---------------
    elif l == 3:
        if debug:
            print('3 vertices')
        shape = cnt.reshape(3, 2)
        a, b, c = shape
        
        # check if it's right angle triangle with a2 + b2 = c2
        l1 = calculatedSquaredDistance(a, b)
        l2 = calculatedSquaredDistance(b, c)
        l3 = calculatedSquaredDistance(c, a)
        v1 = abs(l1 - (l2+l3)) < (0.2*l1)
        v2 = abs(l2 - (l1+l3)) < (0.2*l2)
        v3 = abs(l3 - (l2+l1)) < (0.2*l3)
        if v1 or v2 or v3:
            dif = differentiate2and3(img, accent_masked)

            if dif == 3:
                if v1:
                    top = c
                    bases = [a,b]
                elif v2:
                    top = a
                    bases = [b,c]
                else:
                    top = b
                    bases = [c,a]
                cnt = getStep3Triangle(img, accent_masked, shape, top, bases)
            return dif, cnt

        elif debug:
            print('l1: {}, l2: {}, l3: {}'.format(l1, l2, l3))

    # ---------------
    elif l == 5:
        if debug: 
            print('5 vertices')
        shape = cnt.reshape(5, 2)
        hull = cv2.convexHull(cnt, returnPoints=False)
        if len(hull) == 4:
            dif = differentiate2and3(img, accent_masked)
            if dif == 3:
                x = 10 - hull.sum()  # find out concave vertex index
                a = cnt[(x+1) % 5]  # next to x
                b = cnt[(x+2) % 5]
                c = cnt[(x+3) % 5]
                d = cnt[(x+4) % 5]  # next to x
                l1 = calculatedSquaredDistance(a[0], c[0])
                l2 = calculatedSquaredDistance(b[0], d[0])

                if l1 > l2:  # ac are bases
                    triangle = np.array([b, c, a])
                else:       # bd are bases
                    triangle = np.array([c, b, d])
                return dif, triangle

            return dif, cnt


    # ---------------
    else:
        if debug:
            print('{} vertices'.format(l))
        isStep4 = compareShapes(refcnt_4, cnt, threshold=1.2)
        if isStep4:
            shape = cnt.reshape(l, 2)
            return 4, cnt
        
        # no step matches
        return -1, []

    return -1, []