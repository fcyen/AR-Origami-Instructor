import cv2
import matplotlib.pyplot as plt
import numpy as np
from shapeComparison import prepareImage


image_paths = [
    'assets/sample/IMG_6371.jpg',
    'assets/sample/IMG_6372.jpg',
    'assets/sample/IMG_6373.jpg'
]

for i in range(len(image_paths)):
    img_cnt, img = prepareImage(image_paths[i], original=True)
    window_name = 'Image ' + str(i)
    cv2.imshow(window_name, img)
    # plt.subplot(2,2,i+1)
    # plt.axis('off')
    # plt.imshow(img_gray)

cv2.waitKey(0)

