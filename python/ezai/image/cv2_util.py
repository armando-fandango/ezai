import cv2
import numpy as np


def isbright(image, thresh=0.5):
    # Resize image to 10x10
    # image = cv2.resize(image, (dim, dim))
    # Convert color space to LAB format and extract L channel
    L, A, B = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    # Normalize L channel by dividing all pixel values with maximum pixel value
    L = L / np.max(L)
    # Return True if mean is greater than thresh else False
    return np.mean(L) > thresh


def increase_brightness(image, value=0.01):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # h, s, v = cv2.split(hsv)
    # lim = 255 - value
    # v[v > lim] = 255
    # v[v <= lim] += value
    # final_hsv = cv2.merge((h, s, v))

    hsv[:, :, 2] = cv2.add(hsv[:, :, 2], value)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image
