import numpy as np
import cv2
from matplotlib import pyplot as plt
from shapeComparison import prepareImage


def detectCorners(ref_img, draw_img):
    corners = cv2.goodFeaturesToTrack(ref_img, 25, 0.08, 200)
    corners = np.int0(corners)
    for i in corners:
        x, y = i.ravel()
        cv2.circle(draw_img, (x, y), 3, 255, -1)


#####################################################################
# open the default camera
cap = cv2.VideoCapture(
    'sample_origami_video.mov')

while True:
    success, img = cap.read()
    # imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ==== key controls ====
    keyPressed = cv2.waitKey(1)

    # quit
    if (keyPressed & 0xFF) == ord('q'):
        cap.release()
        cv2.destroyAllWindows()

    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    try:
        img_cnt, img_thresh = prepareImage(img)
        rect = cv2.minAreaRect(img_cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
        cv2.putText(img, str(rect[2]), (100, 100),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    except:
        pass

    cv2.imshow('Image', img)

################################################################


# Useful code snippets
# Blank canvas
# img2 = np.ones((480,640,3))

    # ------ overlay instructions -------
    # note: for drafting purpose only, will be moved somewhere
    # cv2.line(dimg, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0), 1)

    # ref_x1 = approx[1][0][0]
    # ref_x2 = approx[3][0][0]
    # padding = int(abs(ref_x1 - ref_x2)/4)
    # x1 = min(ref_x1, ref_x2) + padding
    # x2 = max(ref_x1, ref_x2) - padding
    # drawCurve(dimg, x1, x2)
