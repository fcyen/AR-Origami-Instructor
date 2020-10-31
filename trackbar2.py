import cv2
import numpy as np

class Trackbar:
    """
    Attributes
    ----------
    trackbars : object
        { trackbarName: [curPos, maxPos] }
    windowName : string
        name of the window
    
    Methods
    -------
    startTrackbars 
        Starts the HSV trackbars

    getTrackbarValues
        Updates the trackbars object with the current positions
    
    closeTrackbars  
        Closes trackbars
    """

    def __init__(self, trackbars, windowName, withMask=0):
        self.windowName = windowName
        self.trackbars = trackbars


    def onChange(self, x):
        pass
    

    def startTrackbars(self):
        wName = self.windowName
        trackbars = self.trackbars

        cv2.namedWindow(wName)
        cv2.resizeWindow(wName, 640, 240)
        for tb in trackbars:
            initPos = trackbars[tb][0]
            maxPos = trackbars[tb][1]
            cv2.createTrackbar(tb, wName, initPos, maxPos, self.onChange)
    

    def getTrackbarValues(self):
        allValues = []

        for tb in self.trackbars:
            curPos = cv2.getTrackbarPos(tb, self.windowName)
            self.trackbars[tb][0] = curPos


    def closeTrackbars(self):
        cv2.destroyWindow(self.windowName)

