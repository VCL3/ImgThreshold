# Project 1: Image Thresholding and Blob Tracking
# Weite Liu & Fengjun Yang

import cv2
import numpy
import sys
import cvk2

# Make a new window named 'Main'.
win = 'Main'
cv2.namedWindow(win)

sources = []

# Open images
if len(sys.argv) > 2:

    num = int(sys.argv[1])

    for n in range(num):
        try:
            sources.append(cv2.imread(sys.argv[n+2]))
        except:
            pass
# If open file unsuccesfully
if len(sources) < num:
    print "Execute main.py followed by the number of images and the corresponding sources"
    print "E.g. python main.py 1 img/purpleflower.jpg"
    sys.exit(1)

# 1. Image Thresholding 
for n in range(num):
    # Take h, w of original image, convert into grayscale
    h = sources[n].shape[0]
    w = sources[n].shape[1]
    display_gray = numpy.empty((h, w), 'uint8')

    cv2.cvtColor(sources[n], cv2.COLOR_RGB2GRAY, display_gray)
    cv2.imwrite(sys.argv[n+2].strip(".jpg")+"_gray.jpg", display_gray)

    # Apply Threshold
    threshold = cv2.adaptiveThreshold(display_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(sys.argv[n+2].strip(".jpg")+"_threshold.jpg", threshold)

# 2. Morphological Operator

