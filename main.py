import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
from sys import platform
import time

from steps import steps, DEBUG
import draw
from shapeMatch import identifyCurrentStep


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
    prevStep = steps[num]
    curStep = steps[num+1]
    count = 0


    # retrieve saved values
    with open('trackbarValues.json') as json_file:
        raw = json.load(json_file)
        hsv = raw[str(0)]
        lowerHSV = np.array(hsv["LowerHSV"])
        upperHSV = np.array(hsv["UpperHSV"])
        hsv_skin = raw[str(2)]   # skin colour
        lowerHSV_skin = np.array(hsv_skin["LowerHSV"])
        upperHSV_skin = np.array(hsv_skin["UpperHSV"])
        hsv_accent = raw[str(3)]  # green colour
        lowerHSV_accent = np.array(hsv_accent["LowerHSV"])
        upperHSV_accent = np.array(hsv_accent["UpperHSV"])

    # -------------------- main loop ------------------------

    while True:
        success, img = cap.read()
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_masked = cv2.inRange(img_hsv, lowerHSV, upperHSV)
        accent_masked = cv2.inRange(img_hsv, lowerHSV_accent, upperHSV_accent)

        if detectHands(img_hsv, lowerHSV_skin, upperHSV_skin):
            # text = 'Hands detected!'
            text = ''
            draw.putInstruction(img, text)

        else:
            if state == 0:
                if num == 0:
                    text1 = 'Place paper on the table to start'
                    text2 = '(white colour side facing up)'
                    draw.putInstruction(img, text1)
                    draw.putInstruction(img, text2, position=(60, 90))

                if prevStep.showNextStep(img, img_masked, accent_masked):  # return True if shape matches
                    if DEBUG:
                        print('showing instructions for step {}'.format(prevStep.id))

                elif curStep.checkShape(img, img_masked, accent_masked):   # return True if shape if confirmed to be correct
                    print('moving to the next step')

                    if num == len(steps)-2:   # last step, proceed to end screen 
                        print("Well done!")
                        state = 1

                    else:
                        num += 1
                        prevStep = steps[num]
                        curStep = steps[num+1]
                
                # elif num != 0 and identifyCurrentStep(img, img_masked, accent_masked, debug=DEBUG)[0] == -1:
                #     draw.putInstruction(img, 'Wrong shape')
                #     draw.putInstruction(img, 'Please try again', position=(60, 90))

            # end screen, shows the wave animation
            elif state == 1: 
                curStep.showNextStep(img, img_masked, accent_masked)

        cv2.imshow('AR Instructor', img)

        keyPressed = cv2.waitKey(1)
        if (keyPressed & 0xFF) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

        # force proceed to next step
        elif (keyPressed & 0xFF) == ord('n'):
            if num < 3:
                num += 1
                prevStep = steps[num]
                curStep = steps[num+1]

            # stop checking previous step
            elif num == 3:
                prevStep = steps[0]
        
        # force retreat to previous step
        elif (keyPressed & 0xFF) == ord('p'):
            if num > 0:
                num -= 1
                prevStep = steps[num]
                curStep = steps[num+1]


def detectHands(img_hsv, l_hsv, u_hsv):
    mask = cv2.inRange(img_hsv, l_hsv, u_hsv)
    h, w = mask.shape
    cropped = mask[h-50:h, 0:w]
    if cropped.sum() > 500000:
        return True
    else:
        return False


main()
