import cv2
import numpy as np

from shapeComparison import detectContour, prepareImage

# prepare images
STEP1 = 'assets/step1.png'
CORRECT1 = 'assets/sample/Correct1.JPG'

ref_img = cv2.imread(STEP1)
ref_img_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
_,ref_img_thresh = cv2.threshold(ref_img_gray, 180, 255, cv2.THRESH_BINARY)
ref_cnt = detectContour(ref_img_thresh)

test_img = cv2.imread(CORRECT1)
test_cnt, test_thresh = prepareImage(CORRECT1)

# --- test image ---
# get bounding rectangle
# returned value: ((x-coordinate, y-coordinate),(width, height), rotation)
rect = cv2.minAreaRect(test_cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(test_img,[box],0,(0,0,255),2)

# rotation
theta = rect[2]
center = rect[0]
shape = (test_img.shape[1], test_img.shape[0]) # cv2.warpAffine expects shape in (length, height)
matrix = cv2.getRotationMatrix2D(center=center, angle=theta, scale=1)
#test_rotated = cv2.warpAffine(src=test_img, M=matrix, dsize=shape)

# perspective warp
width = int(rect[1][0])
height = int(rect[1][1])
src_pts = box.astype("float32")
dst_pts = np.array([[0, height-1],
                    [0, 0],
                    [width-1, 0],
                    [width-1, height-1]], dtype="float32")
matrix_ppt = cv2.getPerspectiveTransform(src_pts, dst_pts)
test_warped = cv2.warpPerspective(test_thresh, matrix_ppt, (width, height))

# --- reference image ---
# get bounding rectangle
ref_rect = cv2.minAreaRect(ref_cnt)
ref_box = cv2.boxPoints(ref_rect)
ref_box = np.int0(ref_box)

# perspective warp
ref_w = int(ref_rect[1][0])
ref_h = int(ref_rect[1][1])
ref_src_pts = ref_box.astype("float32")
ref_dst_pts = np.array([[0, ref_h-1],
                    [0, 0],
                    [ref_w-1, 0],
                    [ref_w-1, ref_h-1]], dtype="float32")
ref_matrix_ppt = cv2.getPerspectiveTransform(ref_src_pts, ref_dst_pts)
ref_warped = cv2.warpPerspective(ref_img_thresh, ref_matrix_ppt, (ref_w, ref_h))

ref_warped_rotated = cv2.rotate(ref_warped, cv2.ROTATE_90_CLOCKWISE)
test_warped_resized = cv2.resize(test_warped, ref_warped.shape) # use unrotated shape because the dimension is inversed

sum_img = cv2.bitwise_xor(test_warped_resized, ref_warped_rotated)
# find min sum
print(np.sum(sum_img))

# rotation
# shape_x = ref_warped.shape[0]
# shape_y = ref_warped.shape[1]
# center_ref = (int(shape_x/2), int(shape_y/2))
# matrix_ref = cv2.getRotationMatrix2D(center=center_ref, angle=90, scale=1)
# ref_warped_rotated = cv2.warpAffine(ref_warped, matrix_ref, (shape_x, shape_y))

#cv2.imshow('Image', test_img)
cv2.imshow('Test', test_warped_resized)
cv2.imshow('Ref', ref_warped_rotated)
cv2.imshow('Sum', sum_img)
cv2.waitKey(0)