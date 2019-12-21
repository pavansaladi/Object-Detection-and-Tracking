
# Import python libraries
import numpy as np
import cv2
#import pdb

# set to 1 for pipeline images
debug = 0


class Detectors(object):
    """Detectors class to detect objects in video frame
    Attributes:
        None
    """
    def __init__(self):
        #'self' is used to represent the instance of a class. 
        #By using the "self" keyword we access the attributes and methods of the class in python.
        #__init__ is a constructor in object oriented terminology
        """Initialize variables used by Detectors class
        Args:
            None
        Return:
            None
        """
        self.fgbg = cv2.createBackgroundSubtractorMOG2()

    def Detect(self, frame):
        """Detect objects in video frame using following pipeline
            - Convert captured frame from BGR to GRAY
            - Perform Background Subtraction
            - Detect edges using Canny Edge Detection
              http://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html
            - Retain only edges within the threshold
            - Find contours
            - Find centroids for each valid contours
        Args:
            frame: single video frame
        Return:
            centers: vector of object centroids in a frame
        """

        # Convert BGR to GRAY
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #reducing complexity from a 3d value to a 1d value

#        if (debug == 0):
#            cv2.imshow('gray', gray)

        # Perform Background Subtraction
        fgmask = self.fgbg.apply(gray)
        # difference of two frames in the video sequence; if this difference
        #exceeds a given threshold, the pixel is considered to be foreground
        
#        if (debug == 0):
#            cv2.imshow('bgsub', fgmask)

        # Detect edges
        edges = cv2.Canny(fgmask, 50, 190, 3)
        #based on the pixel intensity values white(255) and black(0)

        if (debug == 1):
            cv2.imshow('Edges', edges)

        # Retain only edges within the threshold
        ret, thresh = cv2.threshold(edges, 127, 255, 0)

        # Find contours
        contours,img = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #stores the  (x,y) coordinates of the boundary of a shape
        #cv2.RETR_EXTERNAL – retrieves only the extreme outer contours
        #cv2.CHAIN_APPROX_SIMPLE – stores only the corner points
        
        if (debug == 0):
            cv2.imshow('thresh', thresh)

        centers = []  # vector of object centroids in a frame
        # we only care about centroids with size of bug in this example
        # recommended to be tunned based on expected object size for
        # improved performance
        blob_height_thresh = 16

        # Find centroid for each valid contours
        for cnt in contours:
            try:
                # get the bounding rect
                x, y, width, height = cv2.boundingRect(cnt)
                 
                # get the min area rect
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                
                box = np.int0(box)
                # draw a red 'nghien' rectangle
                #cv2.drawContours(img, [box], -1, (255,255,0),1)
                # Calculate and draw circle
                #(x1, y1), radius = cv2.minEnclosingCircle(cnt)
                centeroid = (int(x), int(y))
                #radius = int(radius)
                if (height > blob_height_thresh):
                    #cv2.circle(frame, centeroid, radius, (255, 0,0), 2)
                    cv2.rectangle(frame, centeroid, (x+width, y+height), (255,0,0), 2)
                    b = np.array([[x], [y]])
                    centers.append(np.round(b))
                    #cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
            except ZeroDivisionError:
                pass

        # show contours of tracking objects
        #cv2.imshow('Trackingbus', frame)

        return centers
