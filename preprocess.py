import numpy as np
import cv2
from tensorflow._api.v2.image import image_gradients

# Remove salt and pepper noise
def deblur(img, intensity):
    if intensity == 0:
        return img

    if intensity == 1:
        img = cv2.medianBlur(img, 3)
    elif intensity == 2:
        img = cv2.medianBlur(img, 5)
        img = cv2.medianBlur(img, 7)
    elif intensity == 3:
        img = cv2.medianBlur(img, 3)
        img = cv2.medianBlur(img, 5)
        img = cv2.medianBlur(img, 7)
    else:
        raise Exception("Invalid deblur intensity")

    return img

# Perform edge detection on binary image using morphological gradient
def detect_edges(img, threshold=150):
    kernel = np.ones((5, 6), np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dilated = cv2.dilate(gray, kernel, iterations=1)
    diff = cv2.absdiff(dilated, gray)
    (T, edges) = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Perform thinning using Guo-Hall algorithm
    edges = 255 * thin_edges(np.array(edges // 255))
    edges = cv2.bitwise_not(edges)

    return edges

# Detect edges in an image using HED
def detect_edges_dnn(img):
    return img

# Perform thinning using the Guo-Hall thinning algorithm
def thin_edges(src):
    def iteration(src, iter):
        marker = np.ones(src.shape, np.uint8)
        h, w = src.shape
        changed = 0
        for j, i in np.transpose(np.nonzero(src)):
            if i == 0 or i == w - 1:
                continue
            if j == 0 or j == h - 1:
                continue
            assert src.item(j, i) != 0
            p2 = src.item((j, i - 1))
            p3 = src.item((j + 1, i - 1))
            p4 = src.item((j + 1, i))
            p5 = src.item((j + 1, i + 1))
            p6 = src.item((j, i + 1))
            p7 = src.item((j - 1, i + 1))
            p8 = src.item((j - 1, i))
            p9 = src.item((j - 1, i - 1))
            C = (
                (~p2 & (p3 | p4))
                + (~p4 & (p5 | p6))
                + (~p6 & (p7 | p8))
                + (~p8 & (p9 | p2))
            )
            N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8)
            N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9)
            N = min(N1, N2)
            if iter == 0:
                m = p8 & (p6 | p7 | ~p9)
            else:
                m = p4 & (p2 | p3 | ~p5)
            if C == 1 and 2 <= N <= 3 and m == 0:
                marker[(j, i)] = 0
                changed += 1
        return src & marker, changed

    dst = src.copy()
    i = 0
    while True:
        i += 1
        dst, changed = iteration(dst, 0)
        dst, changed2 = iteration(dst, 1)

        d = changed + changed2
        if d == 0:
            break

        # Safegaurd against infinite loop
        if i == 25:
            break

    return dst


