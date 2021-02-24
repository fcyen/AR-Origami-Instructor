import cv2
import time
import os
import numpy as np
import math

CROP_X = 400
CROP_Y = 150
COLOR = (255, 0, 0)


def convertToIntPoint(point):
    return (int(point[0]), int(point[1]))


def rotate(point, center, angle):
    angle = angle/180 * math.pi  # convert to radians
    x1, y1 = point
    xc, yc = center
    x2 = ((x1 - xc) * math.cos(angle)) - ((y1 - yc) * math.sin(angle)) + xc
    y2 = ((x1 - xc) * math.sin(angle)) + ((y1 - yc) * math.cos(angle)) + yc
    return convertToIntPoint((x2, y2))


def scale(point, center, factor):
    x1, y1 = point
    xc, yc, = center
    x2 = factor * x1 + (1-factor) * xc
    y2 = factor * y1 + (1-factor) * yc
    return convertToIntPoint((x2, y2))



def detectContour(img, disp_img=[], offset=(0, 0)):
    ''' Returns first contour detected in the image'''
    # draw boundaries
    if offset[0] != 0:
        width, height = img.shape
        cv2.rectangle(disp_img, (CROP_X, CROP_Y),
                      (height+CROP_X, width+CROP_Y), (0, 0, 255), 5)

    img_blur = cv2.GaussianBlur(img, (3, 3), 1)
    contours, _ = cv2.findContours(
        img_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = []

    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        # print(area)
        if area > 10000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.005*peri, True)
            l = len(approx)
            result.append(approx.reshape(l, 2))

            if len(disp_img) > 0:
                cv2.drawContours(disp_img, [approx],
                                 0, (255, 0, 0), 10, offset=offset)
                cv2.putText(disp_img, str(
                    area), (approx[0][0][0]+50, approx[0][0][1]+50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

                # %% to label contour points %%
                for i in range(len(approx)):
                    cv2.putText(disp_img, str(
                        i), (approx[i][0][0]+10, approx[i][0][1]+10), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 0), 5)
                cv2.imshow('Image', disp_img)
                cv2.waitKey(0)

    if len(result) > 0:
        return result[0]    # assume first one is the correct one
    else:
        return result


def compareMatchShapes(cnt1, cnt2):
    d4 = d5 = d6 = -1

    if len(cnt1) > 0 and len(cnt2) > 0:
        d4 = cv2.matchShapes(cnt1, cnt2, cv2.CONTOURS_MATCH_I1, 0)
        d5 = cv2.matchShapes(cnt1, cnt2, cv2.CONTOURS_MATCH_I2, 0)
        d6 = cv2.matchShapes(cnt1, cnt2, cv2.CONTOURS_MATCH_I3, 0)

    print('d1: {}  \t, d2: {}\t, d3: {}'.format(d4, d5, d6))


def load_paths_from_folder(folder):
    paths = []
    for filename in os.listdir(folder):
        paths.append(os.path.join(folder, filename))

    return paths, filename


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def prepareImage(img, original=False):
    ''' 
    Returns a contour detected and the binary image 
    @param gray: returns gray image instead of binary 
    '''
    if isinstance(img, str):    # if image path given
        img = cv2.imread(img)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_thresh = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY)

    img_cnt = detectContour(img_thresh, img)

    if original:
        return img_cnt, img
    else:
        return img_cnt, img_thresh


def fromTestSets(folder):
    ref_img = cv2.imread('assets/step4.png', cv2.IMREAD_GRAYSCALE)
    cnt_ref = detectContour(ref_img)

    for filename in os.listdir(folder):
        if filename[0] != '.':
            path = os.path.join(folder, filename)
            cnt, _ = prepareImage(path)
            print(filename, end=',\t')
            compareMatchShapes(cnt_ref, cnt)
        # try:
        #     cnt = prepareImage(path)
        #     print(filename, end=',\t')
        #     compareMatchShapes(cnt_ref, cnt)
        # except:
        #     print(filename)


def fromWebCam():
    # --- load reference shape ---
    ref_img = cv2.imread('assets/new_step4.png')
    ref_img_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    cnt_ref = detectContour(ref_img_gray, ref_img)

    # start camera
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('step4_sample.mov')

    i = 0
    while True:
        success, img = cap.read()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, img_thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

        width, height = img_gray.shape
        # cnt_img = detectContour(
        #     img_thresh[CROP_Y:width-CROP_Y, CROP_X:height-CROP_X], img, (CROP_X, CROP_Y))
        cnt_img = detectContour(img_thresh)

        # quit
        keyPressed = cv2.waitKey(60)
        if (keyPressed & 0xFF) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()

        # d1 = cv2.matchShapes(img_gray, ref_img, cv2.CONTOURS_MATCH_I1, 0)
        # d2 = cv2.matchShapes(img_gray, ref_img, cv2.CONTOURS_MATCH_I2, 0)
        # d3 = cv2.matchShapes(img_gray, ref_img, cv2.CONTOURS_MATCH_I3, 0)

        if len(cnt_ref) > 0 and len(cnt_img) > 0:
            cv2.drawContours(img, [cnt_img], 0, (255, 0, 0), 3)
            d4 = cv2.matchShapes(cnt_ref, cnt_img, cv2.CONTOURS_MATCH_I1, 0)
            d5 = cv2.matchShapes(cnt_ref, cnt_img, cv2.CONTOURS_MATCH_I2, 0)
            d6 = cv2.matchShapes(cnt_ref, cnt_img, cv2.CONTOURS_MATCH_I3, 0)

        cv2.putText(img, str(d4), (100, 100),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        cv2.imshow("Webcam", img)
        cv2.imshow("Binary", img_thresh)

        if d4 < 0.06:
            cv2.waitKey(0)

        # if i == 20:  # slow down printing rate
        #     # print('d1: {}  \t|| d2: {}\t|| d3: {}'.format(d4, d5, d6))
        #     print('d1: {}'.format(d4))
        #     i = 0

        i += 1


def shapeCompare():
    pass


FOLDER = "/Users/foo_c/Documents/Working Folders/FYP/fyp_opencv/assets/sample"
FOLDER_a = "/Users/foo_c/Documents/Working Folders/FYP/fyp_opencv/assets/sample/step4"
# fromTestSets(FOLDER_a)
# fromWebCam()


# img = cv2.imread('assets/step1.png')
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cnt = detectContour(img_gray)
# rect = cv2.minAreaRect(cnt)
# box = cv2.boxPoints(rect)
# box = np.int0(box)
# cv2.drawContours(img,[box],0,(0,0,255),2)
# print(rect[1])


# another way is to create a png then just scale that

def draw(img, center, angle=0, factor=1):
    xc, yc = center
    pt1 = (int(xc - 316/2), int(yc))
    pt2 = (int(xc + 316/2), int(yc))

    pt1r = rotate(pt1, center, angle)
    pt2r = rotate(pt2, center, angle)

    pt1s = scale(pt1, center, factor)
    pt1s = rotate(pt1s, center, angle)
    pt2s = rotate(pt2, center, angle)

    pt1sh = (pt1s[0], pt1s[1]+20)
    pt2sh = (pt2s[0], pt2s[1]+20)

    cv2.line(img, pt1, pt2, (255, 0, 0), 3)
    cv2.line(img, pt1r, pt2r, (0, 255, 0), 3)
    cv2.line(img, pt1sh, pt2sh, (0, 255, 255), 3)
    cv2.circle(img, convertToInt(center), 5, (255, 0, 0), -1)

# cv2.imshow('Image', img)
# cv2.waitKey(0)
