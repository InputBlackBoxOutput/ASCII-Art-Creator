import cv2
import numpy as np

# Resize the image maintaining the aspect ratio
def resize(img, width=0):
    if width == 0:
        width = img.shape[1]

    # Calculate new height maintaining aspect ratio
    aspect_ratio = img.shape[0] / img.shape[1]
    height = int(width * aspect_ratio)

    return cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)


# Add a calculated padding to the image to help convolution operations
def margin(img):
    input_shape = [64, 64, 1]
    margin = (input_shape[0] - 18) // 2

    # Caluculate new heigth and width
    height = img.shape[0] + 2 * margin + 18
    width = img.shape[1] + 2 * margin + 18

    canvas = np.ones((height, width), dtype=np.uint8) * 255
    canvas[margin : margin + img.shape[0], margin : margin + img.shape[1]] = img

    return canvas
