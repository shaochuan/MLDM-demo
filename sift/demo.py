#!/usr/bin/python
import cv, cv2
import im
import sys

gMainWindowName = 'Main Window'
gImageSize = (500, 375)
gMaxDist = 800

class ImageObject(object):
    def __init__(self, name, size=gImageSize):
        super(ImageObject, self).__init__()
        self.name = name
        self.size = size
        self._image = None
        self._keypoints = None
        self._descriptors = None
    @property
    def iplimage(self):
        if not self._image:
            _image = cv.LoadImage(self.name, cv.CV_LOAD_IMAGE_COLOR)
            self._image = im.resize(_image, self.size)
        return self._image
    @property
    def keypoints(self):
        if not self._keypoints:
            self._keypoints, self._descriptors = im.extract_sift(self.iplimage)
        return self._keypoints
    @property
    def descriptors(self):
        if not self._descriptors:
            self._keypoints, self._descriptors = im.extract_sift(self.iplimage)
        return self._descriptors
    @property
    def width(self):
        return self.iplimage.width
    @property
    def height(self):
        return self.iplimage.height
        
def match(imgobj1, imgobj2):
    m = cv2.DescriptorMatcher_create('BruteForce-L1')
    matches = m.match(imgobj1.descriptors, imgobj2.descriptors)
    cv.NamedWindow(gMainWindowName, cv.CV_WINDOW_AUTOSIZE)

    stacked_image = im.stitch_stacking(imgobj1.iplimage, imgobj2.iplimage)

    # draw dots at keypoints.
    for kpt in imgobj1.keypoints:
        cv.Circle(stacked_image, tuple(map(int, kpt.pt)), 1, im.color.green)
    for kpt in imgobj2.keypoints:
        pt = map(int, kpt.pt)
        pt[1] += imgobj1.iplimage.height
        cv.Circle(stacked_image, tuple(pt), 1, im.color.green)

    # draw lines between matches
    for m in matches:
        kpt1 = imgobj1.keypoints[m.queryIdx]
        kpt2 = imgobj2.keypoints[m.trainIdx]
        pt1 = tuple(map(int, kpt1.pt))
        pt2 = map(int, kpt2.pt)
        pt2[1] += imgobj1.height
        pt2 = tuple(pt2)
        if m.distance < gMaxDist:
            cv.Line(stacked_image, pt1, pt2, im.color.green, thickness=1)

    cv.ShowImage(gMainWindowName, stacked_image)

    print "Press 'q' to leave...(focus is on the window)"
    while True:
        k = cv.WaitKey(3)
        k = chr(k) if k > 0 else 0
        if k == 'q':
            break
def isfloat(str):
    try:
        float(str)
        return True
    except ValueError:
        return False

def print_helper():
    print '''
    Usage:

    ./demo.py <filename1.png> <filename2.png>

    or

    ./demo.py <filename1.png> <scale=0.5>
'''
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print_helper()
        sys.exit(0)
    elif isfloat(sys.argv[2]):
        scale = float(sys.argv[2])
        scaled_size = map(int, (gImageSize[0] * scale, gImageSize[1] * scale))
        imgobj2 = ImageObject(sys.argv[1], scaled_size)
    else:
        imgobj2 = ImageObject(sys.argv[2])
    imgobj1 = ImageObject(sys.argv[1])
    match(imgobj1, imgobj2)
