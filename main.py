import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
from sys import platform
import time

from allSteps import steps, DEBUG
import draw


def main():
    if platform == "win32":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0, 1200)
    # cap = cv2.VideoCapture('videos/full_sample.mov')  # use sample video

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_EXPOSURE, -7)  # set exposure to minimum

    state = 0
    num = 0
    step = steps[num]

    # retrieve saved values
    with open('trackbarValues.json') as json_file:
        raw = json.load(json_file)
        hsv = raw[str(0)]
        lowerHSV = np.array(hsv["LowerHSV"])
        upperHSV = np.array(hsv["UpperHSV"])
        hsv_skin = raw[str(2)]
        lowerHSV_skin = np.array(hsv_skin["LowerHSV"])
        upperHSV_skin = np.array(hsv_skin["UpperHSV"])

    # -------------------- main loop ------------------------

    while True:
        success, img = cap.read()
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_masked = cv2.inRange(img_hsv, lowerHSV, upperHSV)
        img_skin = cv2.inRange(img_hsv, lowerHSV_skin, upperHSV_skin)

        if state == 0:
            if num == 0:
                text1 = 'Place paper on the table with'
                text2 = 'the white colour side facing up.'
                draw.putInstruction(img, text1)
                draw.putInstruction(img, text2, position=(60, 90))


            if step.checkShape(img_masked, img):
                num += 1
                state = 1

        elif state == 1:
            hand_detected = img_skin.sum() > 10000000
            if step.showNextStep(img, img_masked) or hand_detected:
                state = 0
                if num == len(steps):
                    print("Well done!")
                    step.showNextStep(img, img_masked)
                    time.sleep(2)
                    cap.release()
                    cv2.destroyAllWindows()
                    break
                else:
                    step = steps[num]

        print("State: {}, step {}".format(state, step.id))

        cv2.imshow('Webcam', img)

        keyPressed = cv2.waitKey(1)
        if (keyPressed & 0xFF) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


main()
