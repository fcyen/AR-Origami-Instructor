import cv2
import numpy as np

def detectShape(img, mask):
    # remove noise
    kernel = np.ones((5, 5), np.uint8)
    cv2.erode(mask, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 400:
            approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
            cv2.drawContours(img, [approx], 0, (255,0,0), 2)    