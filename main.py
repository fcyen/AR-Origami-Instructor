import cv2
import numpy as np
import trackbar

lowerHSV = np.array([0,0,0])
upperHSV = np.array([0,0,0])

def startWebcam(opt):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # open the default camera
    cap.set(3, 640)  # CV_CAP_PROP_FRAME_WIDTH
    cap.set(4, 480)  # CV_CAP_PROP_FRAME_HEIGHT
    cap.set(10, 150)  # CV_CAP_PROP_BRIGHTNESS

    trackbarOn = False

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
                trackbar.initTrackbars(TrackbarChange)
                trackbarOn = True
            else:
                values = trackbar.closeTrackbars()
                trackbarOn = False
        # =======================
        
        if trackbarOn:
            cv2.imshow("Mask", mask)
            #trackbar.useHSVTrackbars(img)

        # === shape detection ===
        # remove noise 
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 400:
                approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
                cv2.drawContours(img, [approx], 0, (255,0,0), 2)

        cv2.imshow("Webcam", img)


class TrackbarChange():
    @staticmethod
    def hueMin(x):
        lowerHSV[0] = x
    
    @staticmethod
    def hueMax(x):
        upperHSV[0] = x

    @staticmethod
    def satMin(x):
        lowerHSV[1] = x
    
    @staticmethod
    def satMax(x):
        upperHSV[1] = x

    @staticmethod
    def valMin(x):
        lowerHSV[2] = x
    
    @staticmethod
    def valMax(x):
        upperHSV[2] = x


startWebcam(1)

