#!/usr/bin/python
import cv
import pdb
gMainWindowName = 'Main Window'

def main():
    cv.NamedWindow(gMainWindowName, cv.CV_WINDOW_AUTOSIZE)
    image = cv.LoadImage('picture.png', cv.CV_LOAD_IMAGE_COLOR)
    
    pdb.set_trace()

if __name__ == '__main__':
    main()
