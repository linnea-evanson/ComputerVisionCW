"""
Script containing function to downsize image to scale given as a percentage of original size

Thomas Bayley
11/02/21
"""
import cv2


def resize(image, scale=50):
    scale_percent = scale  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    return resized
