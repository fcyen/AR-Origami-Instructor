import cv2
import numpy as np
import json
from trackbar import Trackbars
from trackbar2 import Trackbar
import shapeDetection as shape

HSV = 0
CANNY = 1

"""
Press 'q' to quit
      'w' to toggle HSV trackbars
      'e' to toggle Canny threshold trackbars
"""

def startWebcam():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # open the default camera
    cap.set(3, 640)  # CV_CAP_PROP_FRAME_WIDTH
    cap.set(4, 480)  # CV_CAP_PROP_FRAME_HEIGHT
    cap.set(10, 150)  # CV_CAP_PROP_BRIGHTNESS

    # retrieve saved values
    with open('trackbarValues.json') as json_file:
        raw = json.load(json_file)
        hsv = raw[str(HSV)]
        lowerHSV = np.array(hsv["LowerHSV"])
        upperHSV = np.array(hsv["UpperHSV"])
        canny = raw[str(CANNY)]

    # for trackbars
    trackbarOn = [False, False] # [HSV, CANNY]
    tb = Trackbars(lowerHSV, upperHSV)
    cannyTb = Trackbar(canny, "Canny Thresholds")


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
        
        # HSV trackbar
        elif (keyPressed & 0xFF) == ord('w'):
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
            cv2.imshow("Mask", mask)
        if trackbarOn[CANNY]:
            cannyTb.getTrackbarValues()

        # Detection operations
        img2 = np.ones((480,640,3))
        shape.detectShape(img, mask, img2)
        shape.detectLines(img, canny["Threshold1"][0], canny["Threshold2"][0], img2)

        cv2.imshow("Webcam", img)
        cv2.imshow("Result", img2)
        

startWebcam()

