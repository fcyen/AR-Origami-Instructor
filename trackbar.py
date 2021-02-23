import cv2
import numpy as np

TRACKBAR_WINDOW = "Trackbars"


class Trackbars:
    """
    Attributes
    ----------
    lower : np array
        reference to the lowerHSV array
    upper : np array
        reference to the upperHSV array

    Methods
    -------
    startTrackbars 
        Starts the HSV trackbars. Changes in value in the trackbars will be updated to the lowerHSV and upperHSV arrays

    useHSVTrackbars
        ~ currently not in-use
        Opens a window that displays mask based on trackbar values

    closeTrackbars  
        Closes trackbars and return lower & upper values in separate arrays
        - return values might not be necessary because you can get the same values in the lowerHSV & upperHSV arrays
          passed when creating an instance
    """

    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def startTrackbars(self):
        l = self.lower
        u = self.upper

        cv2.namedWindow(TRACKBAR_WINDOW)
        cv2.resizeWindow(TRACKBAR_WINDOW, 640, 240)
        cv2.createTrackbar("Hue Min", TRACKBAR_WINDOW, l[0], 179, self.hueMin)
        cv2.createTrackbar("Hue Max", TRACKBAR_WINDOW, u[0], 179, self.hueMax)
        cv2.createTrackbar("Sat Min", TRACKBAR_WINDOW, l[1], 255, self.satMin)
        cv2.createTrackbar("Sat Max", TRACKBAR_WINDOW, u[1], 255, self.satMax)
        cv2.createTrackbar("Val Min", TRACKBAR_WINDOW, l[2], 255, self.valMin)
        cv2.createTrackbar("Val Max", TRACKBAR_WINDOW, u[2], 255, self.valMax)

    def useHSVTrackbars(self, img):
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        h_min = cv2.getTrackbarPos("Hue Min", TRACKBAR_WINDOW)
        h_max = cv2.getTrackbarPos("Hue Max", TRACKBAR_WINDOW)
        s_min = cv2.getTrackbarPos("Sat Min", TRACKBAR_WINDOW)
        s_max = cv2.getTrackbarPos("Sat Max", TRACKBAR_WINDOW)
        v_min = cv2.getTrackbarPos("Val Min", TRACKBAR_WINDOW)
        v_max = cv2.getTrackbarPos("Val Max", TRACKBAR_WINDOW)

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(imgHSV, lower, upper)

        cv2.imshow("Mask", mask)

    def closeTrackbars(self):
        h_min = cv2.getTrackbarPos("Hue Min", TRACKBAR_WINDOW)
        h_max = cv2.getTrackbarPos("Hue Max", TRACKBAR_WINDOW)
        s_min = cv2.getTrackbarPos("Sat Min", TRACKBAR_WINDOW)
        s_max = cv2.getTrackbarPos("Sat Max", TRACKBAR_WINDOW)
        v_min = cv2.getTrackbarPos("Val Min", TRACKBAR_WINDOW)
        v_max = cv2.getTrackbarPos("Val Max", TRACKBAR_WINDOW)

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        cv2.destroyWindow(TRACKBAR_WINDOW)
        cv2.destroyWindow("Mask")

        return lower, upper

    # onChange callbacks
    def hueMin(self, x):
        self.lower[0] = x

    def hueMax(self, x):
        self.upper[0] = x

    def satMin(self, x):
        self.lower[1] = x

    def satMax(self, x):
        self.upper[1] = x

    def valMin(self, x):
        self.lower[2] = x

    def valMax(self, x):
        self.upper[2] = x
