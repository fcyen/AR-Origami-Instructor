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

def detectShape2(img, t1, t2):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(imgGray, t1, t2)
    #lines = cv2.HoughLines(edges, 1, np.pi/180, 10)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 25, 10, 100)
    
    cv2.imshow('Gray', edges)
    try:
        for line in lines[0]:
            # rho,theta = line
            # a = np.cos(theta)
            # b = np.sin(theta)
            # x0 = a*rho
            # y0 = b*rho
            # x1 = int(x0+1000*(-b))
            # y1 = int(y0+1000*(a))
            # x2 = int(x0-1000*(-b))
            # y2 = int(y0-1000*(a))
            x1,y1,x2,y2 = line
            cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)
    except:
        # when line is a NoneType
        pass