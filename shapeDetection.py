import cv2
import numpy as np


def detectShape(img, mask, dimg):
    # remove noise
    kernel = np.ones((5, 5), np.uint8)
    cv2.erode(mask, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
            cv2.drawContours(dimg, [approx], 0, (255,0,0), 2)    


def detectLines(img, t1, t2, dimg):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.GaussianBlur(img, (3,3), 0, imgGray)

    # this is to recognize white on white
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
    #dilated = cv2.dilate(imgGray, kernel)

    edges = cv2.Canny(imgGray, t1, t2, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=50, maxLineGap=20)
    
    cv2.imshow('Edges', edges)

    try:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(dimg, (x1,y1), (x2,y2), (0,255,0), 2)
    except:
        # when line is a NoneType
        pass




# img = cv2.imread('vertical-lines.jpg')

# while True:
#     keyPressed = cv2.waitKey(1)
#     # quit 
#     if (keyPressed & 0xFF) == ord('q'):
#         cv2.destroyAllWindows()
#         break
#     cv2.imshow("Img", img)
#     detectShape2(img, 20, 80)