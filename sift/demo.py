#!/usr/bin/python
import cv, cv2
import im
import sys

gMainWindowName = 'Main Window'
gImageSize = (500, 375)
gMaxDist = 800

class ImageObject(object):
    def __init__(self, name, iplimage=None, size=gImageSize):
        super(ImageObject, self).__init__()
        self.name = name if not iplimage else None
        self.size = size
        self._image = iplimage
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
        if self._keypoints is None:
            self._keypoints, self._descriptors = im.extract_sift(self.iplimage)
        return self._keypoints
    @property
    def descriptors(self):
        if self._descriptors is None:
            self._keypoints, self._descriptors = im.extract_sift(self.iplimage)
        return self._descriptors
    @property
    def width(self):
        return self.iplimage.width
    @property
    def height(self):
        return self.iplimage.height

def match(imgobj1, imgobj2):
    ''' Returns matches keypoints '''
    m = cv2.DescriptorMatcher_create('BruteForce-L1')
    matches = m.match(imgobj1.descriptors, imgobj2.descriptors)
    return matches

def generate_stacked_image(matches):
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

    return stacked_image

def show_matching(imgobj1, imgobj2):
    matches = match(imgobj1, imgobj2)
    stacked_image = generate_stacked_image(matches)

    cv.NamedWindow(gMainWindowName, cv.CV_WINDOW_AUTOSIZE)
    cv.ShowImage(gMainWindowName, stacked_image)

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

gTracking = False
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print_helper()
        sys.exit(0)

    if 'track' in sys.argv[1:]:
        cam = cv.CaptureFromCAM(0)
        iplimage = cv.QueryFrame(cam)
        iplimage = im.resize(iplimage, gImageSize)
        cv.Flip(iplimage, None, 1)
        last_iplimage = iplimage
        gTracking = True
    else:
        # show matching results
        if isfloat(sys.argv[2]):
            scale = float(sys.argv[2])
            scaled_size = map(int, (gImageSize[0] * scale, gImageSize[1] * scale))
            imgobj2 = ImageObject(sys.argv[1], size=scaled_size)
        else:
            imgobj2 = ImageObject(sys.argv[2])
        imgobj1 = ImageObject(sys.argv[1])
        show_matching(imgobj1, imgobj2)

    print "Press 'q' to leave...(focus is on the window)"
    while True:
        k = cv.WaitKey(3)
        k = chr(k) if k > 0 else 0
        if k == 'q':
            break
        if gTracking:
            # motion tracking
            iplimage = cv.QueryFrame(cam)
            iplimage = im.resize(iplimage, gImageSize)
            cv.Flip(iplimage, None, 1)
            last_imgobj = ImageObject(None, last_iplimage, gImageSize)
            curr_imgobj = ImageObject(None, iplimage, gImageSize)
            matches = match(last_imgobj, curr_imgobj)

            last_iplimage = iplimage
            for kpt in curr_imgobj.keypoints:
                cv.Circle(iplimage, tuple(map(int, kpt.pt)), 1, im.color.green)

            # draw lines between matches
            for m in matches:
                kpt1 = last_imgobj.keypoints[m.queryIdx]
                kpt2 = curr_imgobj.keypoints[m.trainIdx]
                pt1 = tuple(map(int, kpt1.pt))
                pt2 = map(int, kpt2.pt)
                pt2 = tuple(pt2)
                xabs = abs(pt1[0]-pt2[0])
                yabs = abs(pt1[1]-pt2[1])
                if xabs < 20 and yabs < 20:
                    cv.Line(iplimage, pt1, pt2, im.color.green, thickness=1)

            cv.ShowImage(gMainWindowName, iplimage)

