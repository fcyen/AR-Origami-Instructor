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
from shapeDetection import detectShape, findTriangleWithFold, calculatedSquaredDistance

HSV = 0
CANNY = 1
HSV_YELLOW = 2
HSV_WHITE = 3
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

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_EXPOSURE, -7)  # set exposure to minimum

    # retrieve saved values
    with open('trackbarValues.json') as json_file:
        raw = json.load(json_file)
        hsv = raw[str(HSV_YELLOW)]
        lowerHSV = np.array(hsv["LowerHSV"])
        upperHSV = np.array(hsv["UpperHSV"])
        canny = raw[str(CANNY)]

    # for trackbars
    trackbarOn = [False, False]  # [HSV, CANNY]
    tb = Trackbars(lowerHSV, upperHSV)
    cannyTb = Trackbar(canny, "Canny Thresholds")

    state = 3
    t = 0

    while True:
        success, img = cap.read()
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(imgHSV, lowerHSV, upperHSV)

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

        # quit with saving (yellow)
        elif (keyPressed & 0xFF) == ord('t'):
            raw[str(HSV_YELLOW)]["LowerHSV"] = lowerHSV.tolist()
            raw[str(HSV_YELLOW)]["UpperHSV"] = upperHSV.tolist()
            with open('trackbarValues.json', 'w') as json_file:
                json.dump(raw, json_file)

            cap.release()
            cv2.destroyAllWindows()

        # quit with saving (yellow)
        elif (keyPressed & 0xFF) == ord('u'):
            raw[str(HSV_WHITE)]["LowerHSV"] = lowerHSV.tolist()
            raw[str(HSV_WHITE)]["UpperHSV"] = upperHSV.tolist()
            with open('trackbarValues.json', 'w') as json_file:
                json.dump(raw, json_file)

            cap.release()
            cv2.destroyAllWindows()

        # HSV trackbar
        elif (keyPressed & 0xFF) == ord('x'):
            # turn on trackbar
            if not trackbarOn[HSV]:
                tb.startTrackbars()
                trackbarOn[HSV] = True
            else:
                values = tb.closeTrackbars()
                trackbarOn[HSV] = False

        # HSV trackbar (white)
        elif (keyPressed & 0xFF) == ord('w'):
            # turn on trackbar
            if not trackbarOn[HSV]:
                tb.startTrackbars()
                trackbarOn[HSV] = True
            else:
                values = tb.closeTrackbars()
                trackbarOn[HSV] = False

        # HSV trackbar (yellow)
        elif (keyPressed & 0xFF) == ord('y'):
            # turn on trackbar
            if not trackbarOn[HSV]:
                tb.startTrackbars()
                trackbarOn[HSV] = True
            else:
                values = tb.closeTrackbars()
                trackbarOn[HSV] = False

        # Canny trackbar
        elif (keyPressed & 0xFF) == ord('e'):
            if not trackbarOn[CANNY]:
                cannyTb.startTrackbars()
                trackbarOn[CANNY] = True
            else:
                cannyTb.closeTrackbars()
                trackbarOn[CANNY] = False
        # =======================

        # Necessary operations when respective trackbar is on
        if trackbarOn[HSV]:
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
            pt1 = (100, 200)
            r = 200
            x = math.cos(math.radians(t)) * r + 200
            y = math.sin(math.radians(t)) * r + 200
            pt2 = (int(x), int(y))
            pt2 = (600, 300)
            dist = calculatedSquaredDistance(pt1, pt2)
            cv2.line(img_copy, pt1, pt2, draw.DEBUG_GREEN, 2)
            draw.drawWave(img_copy, pt1, pt2, t)
            t += 0.2
            if t > 6.2:
                t = 0
            time.sleep(0.2)


        cv2.namedWindow('Result')
        cv2.imshow("Result", img_copy)


state = 0
startWebcam()


#shape.detectLines(img, canny["Threshold1"][0], canny["Threshold2"][0], img2)
