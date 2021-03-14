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

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_EXPOSURE, -7)  # set exposure to minimum

    state = 0
    num = 0
    curStep = steps[num]
    nextStep = steps[num+1]
    count = 0


    # retrieve saved values
    with open('trackbarValues.json') as json_file:
        raw = json.load(json_file)
        hsv = raw[str(0)]
        lowerHSV = np.array(hsv["LowerHSV"])
        upperHSV = np.array(hsv["UpperHSV"])
        hsv_skin = raw[str(2)]  # skin colour
        lowerHSV_skin = np.array(hsv_skin["LowerHSV"])
        upperHSV_skin = np.array(hsv_skin["UpperHSV"])

    # -------------------- main loop ------------------------

    while True:
        success, img = cap.read()
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_masked = cv2.inRange(img_hsv, lowerHSV, upperHSV)

        if detectHands(img_hsv, lowerHSV_skin, upperHSV_skin):
            # text = 'Hands detected!'
            text = ''
            draw.putInstruction(img, text)

        else:
            if state == 0:
                if num == 0:
                    text1 = 'Place paper on the table'
                    text2 = '(white colour side facing up)'
                    draw.putInstruction(img, text1)
                    draw.putInstruction(img, text2, position=(60, 90))

                if curStep.showNextStep(img, img_masked):  # return True if shape matches
                    print('cur')
                    # if num == 3 or num == 2:
                    #     count += 1

                # return True if shape if confirmed to be correct
                elif nextStep.checkShape(img, img_masked):
                    print('next')
                    # count = 0
                    if num == len(steps)-2:   # last step
                        print("Well done!")
                        state = 1

                    else:
                        num += 1
                        curStep = steps[num]
                        nextStep = steps[num+1]


            elif state == 1:    # end screen
                nextStep.showNextStep(img, img_masked)

        print("State: {}, step {}".format(state, curStep.id))

        cv2.imshow('Webcam', img)

        keyPressed = cv2.waitKey(1)
        if (keyPressed & 0xFF) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

        # force proceed to next step
        elif (keyPressed & 0xFF) == ord('n') or (count > 300):
            if num < (len(steps)-2):
                num += 1
                curStep = steps[num]
                nextStep = steps[num+1]
                count = 0
            elif num == (len(steps)-2):
                curStep = steps[0]


def detectHands(img_hsv, l_hsv, u_hsv):
    mask = cv2.inRange(img_hsv, l_hsv, u_hsv)
    h, w = mask.shape
    cropped = mask[h-50:h, 0:w]
    if cropped.sum() > 1000000:
        return True
    else:
        return False


main()
