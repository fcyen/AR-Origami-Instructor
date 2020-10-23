import cv2
import numpy as np

TRACKBAR_WINDOW = "Trackbars"


def onChange(v):
    pass


def initTrackbars(c):
    cv2.namedWindow(TRACKBAR_WINDOW)
    cv2.resizeWindow(TRACKBAR_WINDOW, 640, 240)
    cv2.createTrackbar("Hue Min", TRACKBAR_WINDOW, 0, 179, c.hueMin)
    cv2.createTrackbar("Hue Max", TRACKBAR_WINDOW, 179, 179, c.hueMax)
    cv2.createTrackbar("Sat Min", TRACKBAR_WINDOW, 0, 255, c.satMin)
    cv2.createTrackbar("Sat Max", TRACKBAR_WINDOW, 255, 255, c.satMax)
    cv2.createTrackbar("Val Min", TRACKBAR_WINDOW, 0, 255, c.valMin)
    cv2.createTrackbar("Val Max", TRACKBAR_WINDOW, 255, 255, c.valMax)


def useHSVTrackbars(img):
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


def closeTrackbars():
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
