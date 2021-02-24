import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapeComparison import prepareImage, detectContour


def getBoundingRect(cnt):
    """ Returns Rect object and coordinates of bounding rectangle """
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return rect, box


def getBoundingImg(img, rect, box):
    """ Returns img cropped along the bounding rectangle """
    width = int(rect[1][0])
    height = int(rect[1][1])
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")
    matrix_ppt = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, matrix_ppt, (width, height))

    # if width > height:
    #     warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    return warped
    

def doAll(img):
    """ Returns cropped binary image and bounding rect """
    img_cnt, img_thresh = prepareImage(img)
    rect, box = getBoundingRect(img_cnt)
    img_warped = getBoundingImg(img_thresh, rect, box)
    return img_warped, rect


image_paths = [
    'assets/sample/IMG_6373.jpg',
    'assets/stepn.png',
    'assets/sample/IMG_6371.jpg',
    'assets/sample/IMG_6372.jpg',
]

# img_cnt, _ = prepareImage(image_paths[0])
# rect, _ = getBoundingRect(img_cnt)
# target_w, target_h = rect[1]


# for i in range(len(image_paths)):
#     result_img = doAll(image_paths[i])

#     if i == 0:
#         w0, h0 = result_img.shape
#     else:
#         # resize the image to match the aspect ratio of the reference image
#         # get size ratio by comparing longer edges of both rectangles
#         w, h = result_img.shape 
#         ratio = max(w0,h0)/max(w,h)
#         result_img = cv2.resize(result_img, (0,0), fx=ratio, fy=ratio)

#     plt.subplot(2,6,i+1)
#     plt.axis('off')
#     plt.title('testing')
#     plt.imshow(cv2.imread(image_paths[i]))
#     plt.subplot(2,6,(i+1)+6)
#     plt.imshow(result_img)

# plt.show()


def contourMap(img1, img2, debug):
    """ Returns the matching orientation of img2 to img1 """
    # rotate images if needed
    w0, h0 = img1.shape
    if w0 < h0:
        img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
        w0, h0 = h0, w0
    if debug:
        plt.subplot(241)
        plt.axis('off')
        plt.imshow(img1)

    w, h = img2.shape
    if w < h:
        print('rotating img2')
        img2 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)

    # first resize img2 to img1's size
    img2 = cv2.resize(img2, (h0, w0))

    # --- comparing different orientation ---
    flip = False
    rotate = False

    # original orientation
    diff = cv2.bitwise_xor(img1, img2)
    min_sum = np.sum(diff)
    if debug:
        plt.subplot(245)
        plt.title(min_sum)
        plt.axis('off')
        plt.imshow(img2)

    # original orientation flipped
    img2f = cv2.flip(img2, 1)
    diff1 = cv2.bitwise_xor(img1, img2f)
    diff_sum1 = np.sum(diff1)
    if diff_sum1 < min_sum:
        min_sum = diff_sum1
        flip = True
    if debug:
        plt.subplot(246)
        plt.title(diff_sum1)
        plt.axis('off')
        plt.imshow(img2f)

    # 180 deg clockwise
    img2b = cv2.rotate(img2, cv2.ROTATE_180)
    diff2 = cv2.bitwise_xor(img1, img2b)
    diff_sum2 = np.sum(diff2)
    if diff_sum2 < min_sum:
        min_sum = diff_sum2
        flip = False
        rotate = True
    if debug:
        plt.subplot(247)
        plt.title(diff_sum2)
        plt.axis('off')
        plt.imshow(img2b)

    # 180 deg clockwise flipped
    img2bf = cv2.rotate(img2f, cv2.ROTATE_180)
    diff3 = cv2.bitwise_xor(img1, img2bf)
    diff_sum3 = np.sum(diff3)
    if diff_sum3 < min_sum:
        flip = True
        rotate = True
    if debug:
        plt.subplot(248)
        plt.title(diff_sum3)
        plt.axis('off')
        plt.imshow(img2bf)
        plt.show()

    # display_img = cv2.addWeighted(img1, 0.5, img2f, 0.5, 0)
    # cv2.imshow('Result', display_img)
    # cv2.waitKey(0)
    
    return (flip, rotate)

live = 3
result_img, result_rect = doAll(image_paths[1])
live_img, live_rect = doAll(image_paths[live])
flip, rotate = contourMap(result_img, live_img, False)


# With the flip and rotate information, transform reference image so that it overlaps nicely on top of feed image
result_img_0 = cv2.imread(image_paths[1])
live_img_0 = cv2.imread(image_paths[live])
box = cv2.boxPoints(live_rect)
box = np.int0(box)
cv2.drawContours(live_img_0,[box],0,(0,0,255),2)
cv2.putText(live_img_0, str(live_rect[2]), (100,100), cv2.FONT_HERSHEY_PLAIN, 10, (0,0,255), 10)
cv2.imshow('bounding box', live_img_0)
cv2.waitKey(0)

# -- 1. flip
# if flip:
#     print('flip')
#     result_img_0 = cv2.flip(result_img_0, 1)

# -- 2. rotate
#   w1 < h1 | no rotate | CW (90-theta)
#   w1 < h1 | rotate    | CCW (90+theta)
#   w1 > h1 | no rotate | CCW theta
#   w1 > h1 | rotate    | CW (180-theta)
# TODO: scaling
(x1, y1), (w1, h1), a1 = live_rect
if w1 < h1:
    if rotate:
        print('1')
        angle = 90+a1
    else:
        print('2')
        angle = -a1
else:
    if rotate:
        print('3')
        angle = a1
    else:
        print('4')
        angle = -(180-a1)

h0, w0, _ = result_img_0.shape
cr = (int(w0/2), int(h0/2))
# angle = -angle
print(angle)
rotation_mat = cv2.getRotationMatrix2D(cr, angle, 1.0)
# find the new width and height bounds
abs_cos = abs(rotation_mat[0,0]) 
abs_sin = abs(rotation_mat[0,1])
bound_w = int(h0 * abs_sin + w0 * abs_cos)
bound_h = int(h0 * abs_cos + w0 * abs_sin)
# subtract old image center (bringing image back to origin) and adding the new image center coordinates
rotation_mat[0, 2] += bound_w/2 - cr[0]
rotation_mat[1, 2] += bound_h/2 - cr[1]
# rotate image with the new bounds and translated rotation matrix
rotated_img = cv2.warpAffine(result_img_0, rotation_mat, (bound_w, bound_h), flags=cv2.INTER_LINEAR)

cv2.imshow("rotated", rotated_img)
# cv2.waitKey(0)

if flip:
    print('flip')
    flipped_img = cv2.flip(rotated_img, 1)

cv2.imshow("flipped", flipped_img)
cv2.waitKey(0)

# -- 3. align centroids
cent_r = result_rect[1]
cent_l = live_rect[1]

# -- 4. blend images
cv2.imshow("rotated", rotated_img)
cv2.waitKey(0)
