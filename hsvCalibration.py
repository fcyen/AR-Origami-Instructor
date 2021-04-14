import cv2
import math
import numpy as np
import json
from trackbar import Trackbars
from trackbar2 import Trackbar
import shapeDetection as shape
import draw
import time
from sys import platform
from shapeDetection import detectShape, findTriangle, findTriangleWithFold, calculatedSquaredDistance, identifyTriangle, differentiateTriangle
from shapeMatch import identifyCurrentStep, detectContours


HSV = 0
CANNY = 1
HSV_A = 2
HSV_B = 3
TEXT_POS = (100, 100)

"""
Press 'q' to quit
      'w' to toggle HSV trackbars
      'e' to toggle Canny threshold trackbars
"""


def startWebcam():
    if platform == 'win32':
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)  # open the default camera

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_EXPOSURE, -7)  # set exposure to minimum

    # retrieve saved values
    with open('trackbarValues.json') as json_file:
        raw = json.load(json_file)
        hsv = raw[str(HSV)]
        lowerHSV = np.array(hsv["LowerHSV"])
        upperHSV = np.array(hsv["UpperHSV"])
        hsv1 = raw[str(HSV_A)]
        l1 = np.array(hsv1["LowerHSV"])
        u1 = np.array(hsv1["UpperHSV"])
        hsv2 = raw[str(HSV_B)]
        l2 = np.array(hsv2["LowerHSV"])
        u2 = np.array(hsv2["UpperHSV"])
        canny = raw[str(CANNY)]

    # for trackbars
    # [HSV, CANNY, HSV_A, HSV_B]
    trackbarOn = [False, False, False, False]
    tb = Trackbars(lowerHSV, upperHSV)
    tb1 = Trackbars(l1, u1)
    tb2 = Trackbars(l2, u2)
    cannyTb = Trackbar(canny, "Canny Thresholds")

    state = 2
    t = 0
    l_hsv = lowerHSV
    u_hsv = upperHSV

    while True:
        success, img = cap.read()
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(imgHSV, l_hsv, u_hsv)
        mask2 = cv2.inRange(imgHSV, l1, u1)
        mask3 = cv2.inRange(imgHSV, l2, u2)

        # ==== key controls ====
        keyPressed = cv2.waitKey(1)
        # quit
        if (keyPressed & 0xFF) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()

        # quit with saving
        elif (keyPressed & 0xFF) == ord('s'):
            raw[str(HSV)]["LowerHSV"] = lowerHSV.tolist()
            raw[str(HSV)]["UpperHSV"] = upperHSV.tolist()
            with open('trackbarValues.json', 'w') as json_file:
                json.dump(raw, json_file)

            cap.release()
            cv2.destroyAllWindows()

        # quit with saving (A)
        elif (keyPressed & 0xFF) == ord('t'):
            raw[str(HSV_A)]["LowerHSV"] = l1.tolist()
            raw[str(HSV_A)]["UpperHSV"] = u1.tolist()
            with open('trackbarValues.json', 'w') as json_file:
                json.dump(raw, json_file)

            cap.release()
            cv2.destroyAllWindows()

        # quit with saving (B)
        elif (keyPressed & 0xFF) == ord('u'):
            raw[str(HSV_B)]["LowerHSV"] = l2.tolist()
            raw[str(HSV_B)]["UpperHSV"] = u2.tolist()
            with open('trackbarValues.json', 'w') as json_file:
                json.dump(raw, json_file)

            cap.release()
            cv2.destroyAllWindows()

        # HSV trackbar
        elif (keyPressed & 0xFF) == ord('x'):
            # turn on trackbar
            if not any(trackbarOn):
                l_hsv = lowerHSV
                u_hsv = upperHSV
                tb.startTrackbars()
                trackbarOn[HSV] = True
            elif trackbarOn[HSV]:
                values = tb.closeTrackbars()
                trackbarOn[HSV] = False
            else:
                print('Close other trackbar first')

        # HSV trackbar (white)
        elif (keyPressed & 0xFF) == ord('z'):
            # turn on trackbar
            if not any(trackbarOn):
                l_hsv = l2
                u_hsv = u2
                tb2.startTrackbars()
                trackbarOn[HSV_B] = True
            elif trackbarOn[HSV_B]:
                values = tb.closeTrackbars()
                trackbarOn[HSV_B] = False
            else:
                print('Close other trackbar first')

        # HSV trackbar (yellow)
        elif (keyPressed & 0xFF) == ord('y'):
            # turn on trackbar
            if not any(trackbarOn):
                l_hsv = l1
                u_hsv = u1
                tb1.startTrackbars()
                trackbarOn[HSV_A] = True
            elif trackbarOn[HSV_A]:
                values = tb.closeTrackbars()
                trackbarOn[HSV_A] = False
            else:
                print('Close other trackbar first')

        # Canny trackbar
        elif (keyPressed & 0xFF) == ord('e'):
            if not any(trackbarOn):
                cannyTb.startTrackbars()
                trackbarOn[CANNY] = True
            elif trackbarOn[CANNY]:
                cannyTb.closeTrackbars()
                trackbarOn[CANNY] = False
            else:
                print('Close other trackbar first')
        # =======================

        # Necessary operations when respective trackbar is on
        if trackbarOn[HSV] or trackbarOn[HSV_B] or trackbarOn[HSV_A]:
            cv2.namedWindow('Mask')
            cv2.imshow("Mask", mask)
        if trackbarOn[CANNY]:
            cannyTb.getTrackbarValues()


# **************************************************************

        img_copy = np.copy(img)

        if state == 0:  # Step 1
            sq1 = []
            try:
                sq1, sq2, sq3, sq4 = shape.findSquare(mask, img_copy)
                if len(sq1) > 0:  # if square is found
                    cv2.line(img_copy, tuple(sq1), tuple(sq3), styles.GREEN, 2)
                    draw.drawCurvedArrow(
                        img_copy, sq2, sq3, sq4, sq1, styles.GREEN)
                    instruction1 = "Fold the paper in half along the green line"
                    cv2.putText(img_copy, instruction1, TEXT_POS,
                                cv2.FONT_HERSHEY_PLAIN, 1)
                    time.sleep(2)
                    state += 1
            except:
                pass
        # check Step 1 completion
        elif state == 1:
            print('2')
            time.sleep(5)
            break

        elif state == 2:    # misc testing
            img_masked = cv2.inRange(imgHSV, lowerHSV, upperHSV)
            accent_masked = cv2.inRange(imgHSV, l2, u2)
            step, shape = identifyCurrentStep(img_copy, img_masked, accent_masked, False)
            print(step)
            

        # cv2.namedWindow('Result')
        cv2.imshow("Webcam", img_copy)
        cv2.imshow("Paper", mask)
        cv2.imshow("Colour accent", mask2)
        cv2.imshow("Skin", mask3)


state = 0
startWebcam()


#shape.detectLines(img, canny["Threshold1"][0], canny["Threshold2"][0], img2)