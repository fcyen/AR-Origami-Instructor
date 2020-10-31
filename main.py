import cv2
import numpy as np
import json
from trackbar import Trackbars
from trackbar2 import Trackbar
import shapeDetection as shape

"""
Press 'q' to quit
      'w' to toggle HSV trackbars
"""

lowerHSV = np.array([0,0,0])
upperHSV = np.array([0,0,0])

houghline = { "Houghline"}

def startWebcam():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # open the default camera
    cap.set(3, 640)  # CV_CAP_PROP_FRAME_WIDTH
    cap.set(4, 480)  # CV_CAP_PROP_FRAME_HEIGHT
    cap.set(10, 150)  # CV_CAP_PROP_BRIGHTNESS

    # retrieve saved values
    with open('trackbarValues.json') as json_file:
        canny = json.load(json_file)

    trackbarOn = False
    tb = Trackbars(lowerHSV, upperHSV)
    cannyTbOn = False
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
            with open('trackbarValues.json', 'w') as json_file:
                json.dump(canny, json_file)

            cap.release()
            cv2.destroyAllWindows()
        
        # HSV trackbar
        elif (keyPressed & 0xFF) == ord('w'):
            # turn on trackbar
            if not trackbarOn:
                tb.startTrackbars()
                trackbarOn = True
            else:
                values = tb.closeTrackbars()
                trackbarOn = False

        # Canny trackbar
        elif (keyPressed & 0xFF) == ord('e'):
            if not cannyTbOn:
                cannyTb.startTrackbars()
                cannyTbOn = True
            else:
                cannyTb.closeTrackbars()
                cannyTbOn = False
        # =======================
        
        if trackbarOn:
            cv2.imshow("Mask", mask)
            #trackbar.useHSVTrackbars(img)
        if cannyTbOn:
            cannyTb.getTrackbarValues()


        #shape.detectShape(img, mask)
        shape.detectShape2(img, canny["Threshold1"][0], canny["Threshold2"][0])

        cv2.imshow("Webcam", img)


startWebcam()

