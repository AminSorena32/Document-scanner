import cv2 as cv
import numpy as np


def Improvement(image, debug=False):
    debug_steps = {}

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if debug:
        debug_steps["1_grayscale"] = gray.copy()

    blur = cv.GaussianBlur(gray, (3, 3), 0)
    if debug:
        debug_steps["2_blurred"] = blur.copy()

    thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv.THRESH_BINARY, 17, 6)
    if debug:
        debug_steps["3_adaptive_threshold"] = thresh.copy()

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=1)

    if debug:
        debug_steps["4_morph_open_final"] = thresh.copy()
        return thresh, debug_steps

    return thresh
