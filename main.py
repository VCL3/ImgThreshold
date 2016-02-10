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
	threshold = cv2.adaptiveThreshold(display_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
	return threshold

def colorThreshold(orig):
	h = orig.shape[0]
	w = orig.shape[1]

	mask = numpy.zeros((h, w), 'uint8')
	orig_float = orig.astype(float)

	# Define the RGB color for the petals of the flower.
	color = [[[34, 217, 0]]]

	# For each pixel in the original image, subtract the petal color.
	dists_float = orig_float - numpy.tile(color, (h, w, 1))

	# Square the differences.
	dists_float = dists_float*dists_float

	# Sum across RGB to get one number per pixel. The result is an array
	dists_float = dists_float.sum(axis=2)

	# Take the square root to get a true distance in RGB space.
	dists_float = numpy.sqrt(dists_float)

	# Allocate space to convert back to uint8, and convert back.
	# This is better than writing
	# 
	#   dists_uint8 = dists_float.astype('uint8')
	#
	# Because it correctly handles overflow (values above 255).
	dists_uint8 = numpy.empty(dists_float.shape, 'uint8')
	cv2.convertScaleAbs(dists_float, dists_uint8, 1, 0)

	# Create a mask by thresholding the distance image at 100.  All pixels
	# with value less than 100 go to zero, and all pixels with value
	# greater than or equal to 100 go to 255.
	cv2.threshold(dists_uint8, 100, 255, cv2.THRESH_BINARY_INV, mask)
	return mask

def opening(img, kernalx = 3, kernaly = 3):
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernalx, kernaly))
	return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def make_mask(img):
	work = img.copy()
	display = numpy.zeros((img.shape[0], img.shape[1], 3),
					  dtype='uint8')
	img, contours, hierarchy = cv2.findContours(work, cv2.RETR_CCOMP,
											  cv2.CHAIN_APPROX_SIMPLE)
	# For each contour in the image
	for j in range(len(contours)):

		# Draw the contour as a colored region on the display image.
		cv2.drawContours(display, contours, j, (255, 255, 255), -1)

		# Compute some statistics about this contour.
		info = cvk2.getcontourinfo(contours[j])

		# Mean location and basis vectors can be useful.
		mu = info['mean']
		b1 = info['b1']
		b2 = info['b2']

	return display

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

# Initialize writer
fps = 30
fourcc, ext = (cv2.VideoWriter_fourcc('D','I','V','X'),'avi')
filename = 'retreived.' + ext
h = frame.shape[0]
w = frame.shape[1]

writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
if not writer:
	print 'Error opening writer'
else:
	print 'Opened', filename, 'for output.'
	writer.write(frame)

# Process every frame
while 1:
	# Get frame
	ok, frame = capture.read(frame)

	# Bail if none
	if not ok or frame is None:
		break

	# 1. Image Thresholding 
	# threshold = imageThreshold(frame)
	threshold = colorThreshold(frame)

	# 2. Morphological Operator
	# morph = opening(threshold)
	
	# 3. Connected Components Analysis + making mask
	# mask = make_mask(morph)
	mask = make_mask(threshold)

	# 4. Refining the mask with another opening
	# mask = opening(mask,4,4)
	bmask = mask.view(numpy.bool)
	display = numpy.zeros((frame.shape[0],frame.shape[1],3),'uint8')
	display[bmask] = frame[bmask]
	cv2.imshow('Video', display)

	# Write if we have a writer.
	if writer:
		writer.write(frame)
	
	cv2.waitKey(5)


