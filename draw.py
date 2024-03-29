import cv2
import math
import numpy as np


# set up a basic canvas here
img = np.zeros(shape=(512, 512, 3), dtype=np.int8)

# ------ constants -------
RED = (0, 0, 255)
# DEBUG_GREEN = (130, 229, 122)
# BLUE = (238, 97, 67)
DEBUG_GREEN = (238, 97, 67)
BLUE = (130, 229, 122)  # changed to green colour for visibility
LIGHTBLUE = (240, 201, 76)
THICKNESS_S = 2
THICKNESS_M = 3

pt_a = (100, 100)
pt_b = (100, 412)
pt_c = (412, 412)
pt_d = (412, 100)


# ====== drawing functions below ======

def drawCurvedArrow(dimg, p1, p2, p3, p4, color=LIGHTBLUE, thickness=THICKNESS_S):
    # move points inwards by 10%
    p1x = round(0.9*p1[0] + 0.1*p3[0])
    p1y = round(0.9*p1[1] + 0.1*p3[1])
    p2x = round(0.9*p2[0] + 0.1*p4[0])
    p2y = round(0.9*p2[1] + 0.1*p4[1])
    p3x = round(0.9*p3[0] + 0.1*p1[0])
    p3y = round(0.9*p3[1] + 0.1*p1[1])
    new_p1 = (p1x, p1y)
    new_p2 = (p2x, p2y)
    new_p3 = (p3x, p3y)

    curve_points = []

    for t in range(100):
        i = t/100
        # formula referenced from: https://youtu.be/pnYccz1Ha34?t=167
        qx = round(((1-i)**2 * new_p1[0] + 2*(1-i)
                    * i*new_p2[0] + i**2 * new_p3[0]))
        qy = round(((1-i)**2 * new_p1[1] + 2*(1-i)
                    * i*new_p2[1] + i**2 * new_p3[1]))
        curve_points.append((qx, qy))

    cv2.polylines(dimg, np.array(
        [curve_points], dtype=np.int32), False, color, thickness=thickness)

    # arrow tip
    drawTriangle(dimg, curve_points[-1], curve_points[-5], color, thickness)


def drawCurvedArrowWithCircle(dimg, p1, p2, p3, p4, color=LIGHTBLUE, thickness=THICKNESS_S):
    new_p1 = (p1, p1)
    new_p2 = (p2, p2)
    new_p3 = (p3, p3)

    curve_points = []

    for t in range(100):
        i = t/100
        # formula referenced from: https://youtu.be/pnYccz1Ha34?t=167
        qx = round(((1-i)**2 * new_p1[0] + 2*(1-i)
                    * i*new_p2[0] + i**2 * new_p3[0]))
        qy = round(((1-i)**2 * new_p1[1] + 2*(1-i)
                    * i*new_p2[1] + i**2 * new_p3[1]))
        curve_points.append((qx, qy))

    cv2.polylines(dimg, np.array(
        [curve_points], dtype=np.int32), False, color, thickness=thickness)
    cv2.circle(dimg, new_p1, 5, color, 3)   # starting point

    # arrow tip
    drawTriangle(dimg, curve_points[-1], curve_points[-5], color, thickness)


def drawTriangle(dimg, v1, vm, color=LIGHTBLUE, thickness=THICKNESS_M):
    # find coordinates
    x1, y1 = v1
    xm, ym = vm
    if ym == y1:
        y1 += 0.01
    m = (x1 - xm)/(ym - y1)
    d2 = (y1 - ym)**2 + (x1 - xm)**2

    X = math.sqrt(d2/(4*(m**2+1)))
    xa = round(X + xm)
    xb = round(-X + xm)
    ya = round(m*X + ym)
    yb = round(-m*X + ym)

    # draw triangle
    vertices = np.array([[x1, y1], [xa, ya], [xb, yb]], np.int32)
    pts = vertices.reshape((-1, 1, 2))
    cv2.polylines(dimg, [pts], isClosed=True, color=color, thickness=thickness)
    cv2.fillPoly(dimg, [pts], color=color)


def putInstruction(dimg, text, scale=0.8, colour=BLUE, thickness=2, position=(60, 60), font=cv2.FONT_HERSHEY_DUPLEX):
    cv2.putText(dimg, text, position, font, scale, colour, thickness)


def drawWave(dimg, pt_a, pt_b, time, color=LIGHTBLUE, thickness=THICKNESS_S):
    wave_points = []
    wave_points_2 = []
    wave_points_3 = []

    amp = 20
    rad = time
    step = 0.02

    xa, ya = pt_a
    xb, yb = pt_b
    mid = ((xa+xb)/2, (ya+yb)/2)
    angle_rad = math.atan(abs(ya-yb)/abs(xa-xb))
    angle = math.degrees(angle_rad)

    for x in range(640):
        y = amp*math.sin(rad)

        pt = (x, y+mid[1])
        pt = rotate(pt, mid, angle)
        wave_points.append(pt)

        pt2 = (x, y+mid[1]+70)
        pt2 = rotate(pt2, (mid[0], mid[1]+70), angle)
        wave_points_2.append(pt2)

        pt3 = (x, y+mid[1]+140)
        pt3 = rotate(pt3, (mid[0], mid[1]+140), angle)
        wave_points_3.append(pt3)

        rad += step
        if rad > 6.28:
            rad = 0

    cv2.polylines(dimg, np.array(
        [wave_points], dtype=np.int32), False, color, thickness=thickness)
    cv2.polylines(dimg, np.array(
        [wave_points_2], dtype=np.int32), False, color, thickness=thickness)
    cv2.polylines(dimg, np.array(
        [wave_points_3], dtype=np.int32), False, color, thickness=thickness)


# =============== Utilities ================

def convertToIntPoint(point):
    return (int(point[0]), int(point[1]))


def rotate(point, center, angle):
    angle = angle/180 * math.pi  # convert to radians
    x1, y1 = point
    xc, yc = center
    x2 = ((x1 - xc) * math.cos(angle)) - ((y1 - yc) * math.sin(angle)) + xc
    y2 = ((x1 - xc) * math.sin(angle)) + ((y1 - yc) * math.cos(angle)) + yc
    return convertToIntPoint((x2, y2))


# unused
def equationroots(a, b, c):
    dis = b * b - 4 * a * c
    sqrt_val = math.sqrt(abs(dis))

    if dis > 0:  # two real roots
        x1 = (-b + sqrt_val)/(2 * a)
        x2 = (-b - sqrt_val)/(2 * a)
        return [x1, x2]

    elif dis == 0:  # one real root
        x = -b / (2 * a)
        return [x, x]

    else:  # complex roots
        return [-1, -1]

# ==========================================

# -- Finding 4th vertex of a square --
# cx = int((pt_a[0]+pt_c[0])/2)
# cy = int((pt_a[1]+pt_c[1])/2)
# center = (cx, cy)
