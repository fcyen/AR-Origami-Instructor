import styles
from shapeComparison import scale, rotate
import allSteps
import matplotlib.pyplot as plt
import numpy as np
import json
import time
import cv2


state = 0
num = 0
steps = allSteps.steps


def main(debug=False):
    cap = cv2.VideoCapture(0, 1200)  # open the default camera
    # cap = cv2.VideoCapture('full_sample.mov')  # use sample video

    state = 0
    num = 0
    step = steps[num]

    # retrieve saved values
    with open('trackbarValues.json') as json_file:
        raw = json.load(json_file)
        hsv = raw[str(0)]
        lowerHSV = np.array(hsv["LowerHSV"])
        upperHSV = np.array(hsv["UpperHSV"])

    while True:
        success, img = cap.read()
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_masked = cv2.inRange(img_hsv, lowerHSV, upperHSV)

        if state == 0:
            if num == 0:
                text = 'Place paper on the table with the white colour side facing up.'
                cv2.putText(img, text, (100, 100),
                            cv2.FONT_HERSHEY_PLAIN, 3, styles.GREEN, 2)

            if step.checkShape(img_masked, img, debug=debug):
                num += 1
                state = 1

        elif state == 1:
            if step.showNextStep(img, img_masked):
                state = 0
                if num == len(steps):
                    print("Well done!")
                    time.sleep(2)
                    cap.release()
                    cv2.destroyAllWindows()
                    break
                else:
                    step = steps[num]

        print("State: {}, step {}".format(state, step.id))

        # retval = cv2.videoio_registry.getCameraBackends()
        # print(retval, cap)  # getBackendName(cap))
        # v = cap.getBackendName(1800)
        cv2.imshow('Webcam', img)

        keyPressed = cv2.waitKey(1)
        if (keyPressed & 0xFF) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


main(debug=True)
