##################################################
# Project 1: Image Thresholding and Blob Tracking
# Weite Liu & Fengjun Yang 
##################################################

import cv2
import numpy
import sys
import cvk2

def cvtThreshold(frame): 
	""" 
		Convert a colored image into grayscale and apply adaptive threshold
	"""
	h = frame.shape[0]
	w = frame.shape[1]
	display_gray = numpy.empty((h, w), 'uint8')

	cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY, display_gray)

	# Apply Threshold
	threshold = cv2.adaptiveThreshold(display_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
	return threshold

def colorThreshold(orig, RGB):
	""" 
		Apply a threshold on the distance of each pixel from the color we want
	"""
	h = orig.shape[0]
	w = orig.shape[1]

	mask = numpy.zeros((h, w), 'uint8')
	orig_float = orig.astype(float)

	# Define the RGB color for the petals of the flower.
	color = [[[RGB[0], RGB[1], RGB[2]]]]

	# Find the difference of colors
	dists_float = orig_float - numpy.tile(color, (h,w,1))
	dists_float = dists_float * dists_float

	# Sum across RGB to get one number per pixel. The result is an array
	dists_float = dists_float.sum(axis=2)

	# Take the square root to get a true distance in RGB space.
	dists_float = numpy.sqrt(dists_float)

	dists_uint8 = numpy.empty(dists_float.shape, 'uint8')
	cv2.convertScaleAbs(dists_float, dists_uint8, 1, 0)
# Create a mask by thresholding the distance image at 100.  All pixels # with value less than 100 go to zero, and all pixels with value
	# greater than or equal to 100 go to 255.
	cv2.threshold(dists_uint8, 50, 255, cv2.THRESH_BINARY_INV, mask)
	return mask

def opening(img, kernalx = 3, kernaly = 3):
	"""
		Apply opening on a given image using an elliptical structuring element
		default parameters for the radius of ellipse are 3 and 3 (a circle)
	"""
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernalx, kernaly))
	return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def make_mask(img):
	"""
		Using findCountour and drawContour to draw a mask for the portion of 
		the image we want
	"""
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

def adaptive(img):
	# 1. Image Thresholding 
	threshold = cvtThreshold(img)

	# 2. Morphological Operator
	morph = opening(threshold,5,5)

	# 3. Connected Components Analysis + making mask
	mask = make_mask(morph)
	
	# 4. Refining the mask with another opening
	mask = opening(mask,6,6)
	bmask = mask.view(numpy.bool)
	display = numpy.zeros((img.shape[0],img.shape[1],3),'uint8')
	display[bmask] = img[bmask]

	return display

def color(img):
	
	# 1.Apply color threshold
	threshold = colorThreshold(img,(30,10,180))
	# 2.Make a mask out of the color threshold
	mask = make_mask(threshold)
	# 3.Get the final image
	bmask = threshold.view(numpy.bool)
	display = numpy.zeros((img.shape[0],img.shape[1], 3),'uint8')
	display[bmask] = img[bmask]

	return display

# Open Video 
input_filename = None

if len(sys.argv) < 3:
	print "Execute main.py followed by the name of input video file\
			and thresholding method(0 for greyscale adaptiveThreshold, 1 for colorThreshold)"
	print "Eg. main.py bunny.mp4"
	sys.exit(1)

elif len(sys.argv) > 2:
	input_filename = sys.argv[1]
	capture = cv2.VideoCapture(input_filename)
	if capture:
		print 'Opened file', input_filename
	# Quit program if failed to open video
	if not capture or not capture.isOpened():
		print 'Error opening video'
		sys.exit(2)
	useAdaptive = sys.argv[2] == '0'

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

	# Process the img
	display = numpy.zeros((frame.shape[0],frame.shape[1], 3),'uint8')
	if useAdaptive:
		display = adaptive(frame)
	else:
		display = color(frame)

	# Project the frame
	cv2.imshow('Video', display)
	cv2.waitKey(5)

	# Write if we have a writer.
	if writer:
		writer.write(display)

