import cv2
import numpy as np

def add_logo(path1, path2, output_path):
    img1 = cv2.imread(path1)
    logo = cv2.imread(path2)

    rows, cols, channels = logo.shape
    roi = img1[0:rows, 0:cols]

    logogray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(logogray, 220, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)

    img1_bg = cv2.bitwise_and(roi, roi, mask = mask_inv)
    logo_fg = cv2.bitwise_and(logo, logo, mask = mask)

    dst = cv2.add(img1_bg, logo_fg)
    img1[0:rows, 0:cols] = dst
    output  = img1

    cv2.imshow('output', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(output_path, output)

add_logo('images/arnold2.jpg', 'images/logo.png', 'images/logo_plus_image.png')