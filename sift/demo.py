#!/usr/bin/python
import cv, cv2
import im
import sys

gMainWindowName = 'Main Window'
gImageSize = (500, 375)

keypoints = {}
descriptors = {}

def add_image_features(name):
    img = cv.LoadImage(name, cv.CV_LOAD_IMAGE_COLOR)
    img = im.resize(img, gImageSize)
    keypts, desctrs = im.extract_sift(img)
    keypoints[img] = keypts
    descriptors[img] = desctrs
    return img

def main(png1, png2):
    img1 = add_image_features(png1)
    img2 = add_image_features(png2)
    max_dist = 1000
    m = cv2.DescriptorMatcher_create('BruteForce-L1')
    matches = m.match(descriptors[img1], descriptors[img2])
    cv.NamedWindow(gMainWindowName, cv.CV_WINDOW_AUTOSIZE)

    stacked_image = im.stitch_stacking(img1, img2)

    # draw dots at keypoints.
    for kpt in keypoints[img1]:
        cv.Circle(stacked_image, tuple(map(int, kpt.pt)), 1, im.color.green)
    for kpt in keypoints[img2]:
        pt = map(int, kpt.pt)
        pt[1] += img1.height
        cv.Circle(stacked_image, tuple(pt), 1, im.color.green)

    # draw lines between matches
    for m in matches:
        kpt1 = keypoints[img1][m.queryIdx]
        kpt2 = keypoints[img2][m.trainIdx]
        pt1 = tuple(map(int, kpt1.pt))
        pt2 = map(int, kpt2.pt)
        pt2[1] += img1.height
        pt2 = tuple(pt2)
        if m.distance < max_dist:
            cv.Line(stacked_image, pt1, pt2, im.color.green, thickness=1)

    cv.ShowImage(gMainWindowName, stacked_image)

    print "Press 'q' to leave...(focus is on the window)"
    while True:
        k = cv.WaitKey(3)
        k = chr(k) if k > 0 else 0
        if k == 'q':
            break

def print_helper():
    print '''
    Usage:

    ./demo.py <filename1.png> <filename2.png>
'''
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print_helper()
        sys.exit(0)
    main(sys.argv[1], sys.argv[2])
