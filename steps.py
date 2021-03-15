import cv2
import math
import time

import draw
from shapeMatch import identifyCurrentStep

convertToIntPoint = draw.convertToIntPoint

steps = []  # array of Step instances
DEBUG = False

class Step:
    def __init__(self, id, draw_fn, instruction, **kwargs):
        '''
        Parameters:
            draw_fn: function - for drawing graphics, must accept bounding rectangle as parameter
            instruction: str - instruction to be displayed
        '''
        self.id = id
        self.draw = draw_fn
        self.instruction = instruction
        self.kwargs = kwargs
        self.count = 0
        self.match_count = 0

    def checkShape(self, img, img_masked, accent_masked, debug=DEBUG):
        step, shape = identifyCurrentStep(img, img_masked, accent_masked, debug)
        shape_match = step == self.id

        # return True after 10 consecutive matches
        if shape_match and self.match_count > 20:
            self.match_count = 0
            return True

        elif shape_match:
            self.match_count += 1
        elif self.match_count > 0:
            self.match_count -= 2
            return False


    def showNextStep(self, dimg, img_masked, accent_masked, debug=DEBUG):
        ''' Displays instruction graphics, returns True if shape still matches '''
        step, cnt = identifyCurrentStep(dimg, img_masked, accent_masked, debug)
        shape_match = step == self.id

        if shape_match:
            if self.id == 4:
                draw.putInstruction(dimg, self.instruction[0])
                # for the wave animation
                self.count += 1
                if self.count > 6:
                    self.count = 0

                self.draw(dimg, cnt, self.count/10)

            else:
                cv2.drawContours(dimg, [cnt], 0, draw.BLUE, draw.THICKNESS_S)
                self.draw(dimg, cnt)

                # instructions
                x = 0
                for instr in self.instruction:
                    draw.putInstruction(
                        dimg, self.instruction[x], position=(60, 60+(30*x)))
                    x += 1
            return True

        return False
 

 # ~~~~~~~~~~~~~~~~~ Step 0 ~~~~~~~~~~~~~~~~~~
step0 = Step(0, None, '')
def dummyFn(self, x,y): 
    return False
step0.showNextStep = dummyFn

# ~~~~~~~~~~~~~~~~~ Step 1 ~~~~~~~~~~~~~~~~~~~
def draw1(img, ref_cnt):
    ref_cnt = ref_cnt.reshape(4, 2)
    pt1 = tuple(ref_cnt[0])
    pt2 = tuple(ref_cnt[1])
    pt3 = tuple(ref_cnt[2])
    pt4 = tuple(ref_cnt[3])
    # draws a line across the diagonal of the square
    cv2.line(img, pt2, pt4, draw.LIGHTBLUE, draw.THICKNESS_M)
    draw.drawCurvedArrow(img, pt1, pt2, pt3, pt4)


instruction1 = ["Fold paper in half"]
step1 = Step(1, draw1, instruction1)


# ~~~~~~~~~~~~~~~~~ Step 2a ~~~~~~~~~~~~~~~~~~~
def draw2(img, ref_cnt):
    # find distance between each vertex
    cnt = ref_cnt.reshape(3, 2)
    (ax, ay), (bx, by), (cx, cy) = cnt
    ab = (ax-bx)**2 + (ay-by)**2
    bc = (bx-cx)**2 + (by-cy)**2
    ac = (ax-cx)**2 + (ay-cy)**2

    long_edge = max(ab, bc, ac)
    if ab == long_edge:
        base1 = (ax, ay)
        base2 = (bx, by)
        top = (cx, cy)
    elif bc == long_edge:
        base1 = (bx, by)
        base2 = (cx, cy)
        top = (ax, ay)
    else:  # ac
        base1 = (ax, ay)
        base2 = (cx, cy)
        top = (bx, by)

    # draw line
    line_start = base1
    r = 0.3827
    line_end = (r*base2[0] + (1-r)*top[0], r*base2[1] + (1-r)*top[1])
    line_end = convertToIntPoint(line_end)
    cv2.line(img, line_start, line_end, draw.LIGHTBLUE, draw.THICKNESS_M)

    # draw curved arrow
    arrow_a = top
    arrow_b = ((base2[0]+top[0])/2, (base2[1]+top[1])/2)
    arrow_c = ((base2[0]+base1[0])/2, (base2[1]+base1[1])/2)
    arrow_d = ((base1[0]+top[0])/2, (base1[1]+top[1])/2)
    draw.drawCurvedArrow(img, arrow_a, arrow_b, arrow_c, arrow_d)


instruction2 = ["Fold the top layer along the",
                 "blue line"]
step2 = Step(2, draw2, instruction2)


# ~~~~~~~~~~~~~~~~~ Step 3a ~~~~~~~~~~~~~~~~~~~
def draw3(img, ref_cnt):
    """ 
    @param ref_cnt: triangle whose vertices are in the order of (top, base1, base2)
    """
    cnt = ref_cnt.reshape(3, 2)
    top, base1, base2 = cnt

    if DEBUG:
        t = tuple(top)
        b1 = tuple(base1)
        b2 = tuple(base2)
        cv2.circle(img, t, 3, (0, 255, 255), -1)
        cv2.putText(img, 'top', t, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255))
        cv2.circle(img, b1, 3, (0, 255, 255), -1)
        cv2.putText(img, 'base1', b1,
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255))
        cv2.circle(img, b2, 3, (0, 255, 255), -1)
        cv2.putText(img, 'base2', b2,
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255))

    # -- draw line
    r1 = 0.2
    r2 = 0.3
    line_start = (r1*base1[0] + (1-r1)*top[0], r1*base1[1] + (1-r1)*top[1])
    line_end = (r2*base1[0] + (1-r2)*base2[0], r2*base1[1] + (1-r2)*base2[1])
    line_start = convertToIntPoint(line_start)
    line_end = convertToIntPoint(line_end)
    cv2.line(img, line_start, line_end, draw.LIGHTBLUE, draw.THICKNESS_S)

    # -- draw curved arrow
    # compute the center of the contour
    M = cv2.moments(ref_cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    center = (cX, cY)
    arrow_a = draw.rotate(base2, center, -20)
    arrow_b = draw.rotate(top, center, -20)
    arrow_c = draw.rotate(base1, center, -20)
    # find 4th vertex
    arrow_dX = arrow_a[0] + arrow_c[0] - arrow_b[0]
    arrow_dY = arrow_a[1] + arrow_c[1] - arrow_b[1]
    arrow_d = (arrow_dX, arrow_dY)
    draw.drawCurvedArrow(img, arrow_a, arrow_b, arrow_c, arrow_d)


instruction3 = ["Fold along the blue line"]
step3 = Step(3, draw3, instruction3)


# ~~~~~~~~~~~~~~~~~ Step 4 ~~~~~~~~~~~~~~~~~~~
def draw4(img, ref_cnt, time):

    if len(ref_cnt) == 7:
        hull = cv2.convexHull(ref_cnt, returnPoints=False)
        if len(hull) == 5:
            tip = (21 - hull.sum())/2
            if tip == 2.5:  # 5 and 0
                tip = 6
            elif tip == 3.5:  # 6 and 1
                tip = 0
            else:
                tip = int(tip)

            pt1 = ref_cnt[(tip+3) % 7][0]
            pt2 = ref_cnt[(tip-3) % 7][0]
            draw.drawWave(img, pt1, pt2, time)

    elif len(ref_cnt) == 6:
        hull = cv2.convexHull(ref_cnt, returnPoints=False)
        if len(hull) == 4:
            tip = (16 - hull.sum())/2
            if tip == 2:  # 5 and 0
                tip = 5
            elif tip == 3:  # 5 and 1
                tip = 0
            else:
                tip = int(tip)

            pt1 = ref_cnt[(tip+3) % 6][0]
            pt2 = ref_cnt[(tip-2) % 6][0]
            draw.drawWave(img, pt1, pt2, time)


instruction4 = ["Origami completed! Well done!!"]
step4 = Step(4, draw4, instruction4)


steps.append(step0)
steps.append(step1)
steps.append(step2)
steps.append(step3)
steps.append(step4)