import cv2
import numpy as np
from trackbar import Trackbars
import shapeDetection as shape

"""
Press 'q' to quit
      'w' to toggle HSV trackbars
"""

lowerHSV = np.array([0,0,0])
upperHSV = np.array([0,0,0])

def startWebcam():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # open the default camera
    cap.set(3, 640)  # CV_CAP_PROP_FRAME_WIDTH
    cap.set(4, 480)  # CV_CAP_PROP_FRAME_HEIGHT
    cap.set(10, 150)  # CV_CAP_PROP_BRIGHTNESS

    trackbarOn = False
    tb = Trackbars(lowerHSV, upperHSV)

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
        
        # trackbar
        elif (keyPressed & 0xFF) == ord('w'):
            # turn on trackbar
            if not trackbarOn:
                tb.startTrackbars()
                trackbarOn = True
            else:
                values = tb.closeTrackbars()
                trackbarOn = False
        # =======================
        
        if trackbarOn:
            cv2.imshow("Mask", mask)
            #trackbar.useHSVTrackbars(img)

        shape.detectShape(img, mask)

        cv2.imshow("Webcam", img)


startWebcam()

