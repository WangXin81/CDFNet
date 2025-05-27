import cv2
import numpy as np


img = cv2.imread("res/test_image_1000.png")
print(np.sum(img))