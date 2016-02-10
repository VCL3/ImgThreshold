##################################################
# Project 1: Image Thresholding and Blob Tracking
# Weite Liu & Fengjun Yang
##################################################

import cv2
import numpy
import sys
import cvk2

def imageThreshold(frame):
    # Take h, w of original image, convert into grayscale
    h = frame.shape[0]
    w = frame.shape[1]
    display_gray = numpy.empty((h, w), 'uint8')

    cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY, display_gray)

    # Apply Threshold
    threshold = cv2.adaptiveThreshold(display_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY, 11, 2)
    cv2.imshow('Video', display_gray)

# Make a new window named 'Main'.
win = 'Main'
# cv2.namedWindow(win)

input_filename = None

# Open Video
if len(sys.argv) < 2:
	print "Execute main.py followed by the name of input video file"
	print "Eg. main.py bunny.mp4"
	sys.exit(1)

elif len(sys.argv) >= 2:
	input_filename = sys.argv[1]
	capture = cv2.VideoCapture(input_filename)
	if capture:
		print 'Opened file', input_filename
	# Quit program if failed to open video
	if not capture or not capture.isOpened():
		print 'Error opening video'
		sys.exit(2)

# Fetch the first frame
ok, frame = capture.read()
if not ok or frame is None:
	print 'No frame in video'
	sys.exit(3)

# Process every frame
while 1:
    # Get frame
	ok, frame = capture.read(frame)

	# Bail if none
	if not ok or frame is None:
		break

	# 1. Image Thresholding 
	imageThreshold(frame)
	# 2. Morphological Operator

	# Apply Threshold
	while cv2.waitKey(15) < 0: pass
"""
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
    print "Execute main.py followed by the number of images and\
		the corresponding sources"
    print "E.g. python main.py 1 img/purpleflower.jpg"
    sys.exit(1)
"""

