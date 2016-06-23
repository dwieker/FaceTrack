import cv2
import numpy as np


# Create a new frame class, derived from the wxPython Frame.
class ControlPanel:
    def __init__(self):
        self.display_names = {
                                "display_scaling": "Display Size (%)",
                                "jerk_threshold": "Face Movement Threshold (%)",
                                "min_img_size": "Face Downscaling Limit",
                                "max_faceless_frames": "Dropped frames limit",
                                "window": "Time Window (s)",
                                "FPS_scaling": "Video Play Speed (%)"
                            }


        cv2.namedWindow("panel")
        cv2.createTrackbar("Display Size (%)", "panel", 80, 200, lambda x: 0)
        cv2.createTrackbar("Face Movement Threshold (%)", "panel", 3, 10, lambda x: 0)
        cv2.createTrackbar("Face Downscaling Limit", "panel", 80, 200, lambda x: 0)
        cv2.createTrackbar("Dropped Frames Limit", "panel", 8, 20, lambda x: 0)
        cv2.createTrackbar("Time Window (s)", "panel", 15, 30, lambda x: 0)
        cv2.createTrackbar("Video Play Speed (%)", "panel", 100, 200, lambda x: 0)


        cv2.imshow("panel", np.ones((1,400)))

    def get(self, setting):
        name = self.display_names[setting]
        if setting in ["FPS_scaling", "jerk_threshold", "display_scaling"]:
            result = cv2.getTrackbarPos(name , "panel") / 100.
        else:
            result = cv2.getTrackbarPos(name , "panel") 

        return result


