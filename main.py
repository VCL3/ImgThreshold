# Project 1: Image Thresholding and Blob Tracking
# Weite Liu & Fengjun Yang

import cv2
import numpy
import sys
import cvk2

# Create filenames for given number of images
def createFiles(num):
    files = []
    for n in range(num):
        filename = sys.argv[n+2].strip(".jpg") + ".txt"
        files.append(filename)
    return files

# This function creates the widget for each image given, loading and/or saving data from
# the filename given.  
def createWidget(widget, filename, showparams, source):

    global win

    if widget.load(filename):
        print 'loaded from', filename
        
    ok = widget.start(win, source)
    
    if ok:
        widget.save(filename)
        if showparams:
            print 'got params: ', widget.params()
        else:
            print 'got points:'
            for p in widget.points:
                print ' ', p

    return widget.points.astype(numpy.float32).reshape((-1, 1, 2))

# Make a new window named 'Main'.
win = 'Main'
cv2.namedWindow(win)

sources = []

# Find the sources of target images
print "Execute main.py followed by the number of images and the corresponding sources"
print "E.g. python main.py 1 homography-images/crop1.jpg"

if len(sys.argv) > 2:

    num = int(sys.argv[1])

    for n in range(num):
        try:
            sources.append(cv2.imread(sys.argv[n+2]))
        except:
            pass

if len(sources) < num:
    print "Please check input format"
    sys.exit(1)

# Create the filenames storing widgets
files = createFiles(num)
# Translated point sets used for homography
point_sets = []
# Homography matrices
homography = []
#find the widget
for i in range(num):
    points = createWidget(cvk2.MultiPointWidget('points'), files[i], False, sources[i])
    point_sets.append(points)

# Find homography destination is the plane of first images
for i in range(num):
    if i == 0:
        homography.append(0)
    else:
        H, mask = cv2.findHomography(point_sets[i], point_sets[0], cv2.RANSAC)
        homography.append(H)

# Compose images into a single scene
for i in range(num):
    if i == 0:
        merged = sources[0]
    else:
        # Get homography matrix
        H = homography[i]
        # Empty array 
        allpoints = numpy.empty( (0, 1, 2), dtype='float32' )

        # Merged specs
        h_orig, w_orig = merged.shape[:2]
        # Get the corner points
        p_orig = numpy.array( [ [[0, 0]],
            [[w_orig, 0]],
            [[w_orig, h_orig]],
            [[0, h_orig]] ], dtype='float32' )
        allpoints = numpy.vstack((allpoints, p_orig))

        # Source image
        h, w = sources[i].shape[:2]
        # Get the corner points
        p = numpy.array( [ [[0, 0]],
                [[w, 0]],
                [[w, h]],
                [[0, h]] ], dtype='float32' )
        # Map through warp
        pp = cv2.perspectiveTransform(p, H)
        allpoints = numpy.vstack((allpoints, pp))
        # Get integer bounding box of form (x0, y0, width, height)
        box = cv2.boundingRect(allpoints)
        dimension = box[2:4]
        p0 = box[0:2]
        # Create Translation Matrix T*H
        T = numpy.eye(3)
        T[0,2] -= p0[0]
        T[1,2] -= p0[1]
        Hnew = numpy.matrix(T) * numpy.matrix(H)
        # Warp two images
        warp_new = cv2.warpPerspective(sources[1], Hnew, dimension)
        warp_orig = cv2.warpPerspective(merged, T, dimension)

        # cv2.imshow('warp_new', warp_new)
        # while cv2.waitKey(15) < 0: pass

        # cv2.imshow('warp_orig', merged)
        # while cv2.waitKey(15) < 0: pass

        merged = warp_new/2 + warp_orig/2

        cv2.imshow('Merged Image', merged)
        while cv2.waitKey(15) < 0: pass

# Write the final image
output_file = sys.argv[2].strip('.jpg') + "_merged.jpg"
cv2.imwrite(output_file, merged)
