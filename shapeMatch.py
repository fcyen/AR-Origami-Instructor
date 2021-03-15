import cv2
import math

import draw

def detectContours(bimg, dimg, minArea=1000, approx_val=0.02, debug=False):
    ''' Returns all approximated contours larger than minArea '''
    contours, _ = cv2.findContours(bimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []

    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)

        if area > minArea:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, approx_val*peri, True)
            result.append(approx)
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


refcnt_4 = loadReferenceContour('assets/step4a.png')

def identifyCurrentStep(img, img_masked, accent_masked, debug=False):
    ''' Returns:
            - step: [int] current step
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
            return 2, cnt
        elif debug:
            print('l1: {}, l2: {}, l3: {}'.format(l1, l2, l3))

    # ---------------
    elif l == 5:
        if debug: 
            print('5 vertices')
        shape = cnt.reshape(5, 2)
        hull = cv2.convexHull(cnt, returnPoints=False)
        if len(hull) == 4:
            return 3, cnt

    # ---------------
    else:
        isStep4 = compareShapes(refcnt_4, cnt)
        if isStep4:
            shape = cnt.reshape(l, 2)
            return 4, cnt
        
        # no step matches
        return -1, []