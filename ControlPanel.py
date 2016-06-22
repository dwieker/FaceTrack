import cv2
import numpy as np


# Create a new frame class, derived from the wxPython Frame.
class ControlPanel:
    def __init__(self):
        cv2.namedWindow("panel")
        cv2.createTrackbar("display_scaling", "panel", 80, 200, lambda x: 0)
        cv2.createTrackbar("jerk_threshold", "panel", 3, 10, lambda x: 0)
        cv2.createTrackbar("min_img_size", "panel", 8000, 20000, lambda x: 0)
        cv2.createTrackbar("max_faceless_frames", "panel", 800, 2000, lambda x: 0)
        cv2.createTrackbar("window", "panel", 1500, 3000, lambda x: 0)


        cv2.imshow("panel", np.ones((1,400)))

    def get(self, setting):
        result = cv2.getTrackbarPos(setting , "panel") / 100.
        return result


