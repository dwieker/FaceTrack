import cv2
import numpy as np
from scipy.misc import imresize, imrotate

class FaceTracker():

    # Initialize Cascade Calssifiers
    faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    eyeCascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")

    def __init__(   self,
                    control_panel,
                    ROI_padding = .2,
                    ROI_speed = .10,
                    eye_n = 1, eye_scaling = 1.2,
                    face_n = 1, face_scaling = 1.1,
                    img_size = 1.0 ):
    

        self.control_panel = control_panel
        self.ROI_padding = ROI_padding 
        self.ROI_speed = ROI_speed 
        self.eye_n = eye_n
        self.eye_scaling = eye_scaling 
        self.face_n = face_n; self.face_scaling = face_scaling
        self.img_size = img_size

        self.ROI = None
        self.face = None
        self.frame = None
        self.consec_facelss_frames = 0
  

    def update_frame(self, frame):
        self.frame = frame
        if self.ROI is None:
            print "Setting ROI to full frame"
            self.ROI = ((0, 0), (frame.shape[1], frame.shape[0]))

        
    def calculate_ROI_around_face(self, x,y,w,h):
        # Input parameters are a FACE. Output is an exanded ROI, accounting for frame boundaries
        ROI_padding = self.ROI_padding
        x1_new, y1_new, x2_new, y2_new =(max(0, x - int(ROI_padding*w)),
                                        max(0, y - int(ROI_padding*h)),
                                        min(self.frame.shape[1], x + int((1+ROI_padding)*w)),
                                        min(self.frame.shape[0], y + int((1+ROI_padding)*h))) 
        return ((x1_new, y1_new), (x2_new, y2_new))

    def expand_ROI(self):
        (x1, y1), (x2, y2) = self.ROI
        ROI_w, ROI_h = x2-x1, y2-y1
        self.ROI =((max(0, x1 - int(self.ROI_speed*ROI_w)), 
                    max(0, y1 - int(self.ROI_speed*ROI_h))),
                   (min(self.frame.shape[1], x2 + int(self.ROI_speed*ROI_w)), 
                    min(self.frame.shape[0], y2 + int(self.ROI_speed*ROI_h)))) 


    def has_moved(self, old, new, threshold):
        (x_old, y_old, w_old, h_old) = old
        (x,y,w,h) = new
        xdiff = abs((x_old + w_old + x_old)/2. -  (x + w + x)/2.)
        ydiff = abs((y_old + h_old + y_old)/2. -  (y + h + y)/2.)
        return (xdiff > threshold) or (ydiff > threshold)


    def locate_face(self):

        # Cut out region of interest to reduce computational time. 
        (x1, y1), (x2, y2) = self.ROI
        gray = cv2.cvtColor(self.frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

        gray = imresize(gray, self.img_size)

        # Detect faces in the gray image
        faces = FaceTracker.faceCascade.detectMultiScale(
            gray,
            scaleFactor=self.face_scaling,
            minNeighbors=self.face_n,
            minSize=(30, 30),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )


        # Initially assume no face was found. Now, search through candidate faces
        face_found = False
        for (x, y, w, h) in faces:
    
            # Check for eyes! If no eyes, no face!
            eyes = FaceTracker.eyeCascade.detectMultiScale(
                gray[y:y+h, x:x+w],
                scaleFactor=self.eye_scaling,
                minNeighbors=self.eye_n,
                minSize=(int(w*.1), int(h*.1)), #Eyes should not be tiny compared to face
            )
            
            # Were Eyes found?
            if len(eyes) > 0: 
      
                x = int(x / self.img_size); y = int(y / self.img_size)
                w = int(w / self.img_size); h = int(h / self.img_size)
                x += x1; y += y1;

                # If face size is large, we can shrink the image to speed up comp
                self.img_size = self.control_panel.get("min_img_size") / h
               
                # Reset faceless frame count
                self.consec_facelss_frames = 0
            
                # If there was a face last frame, just use that position again
                # UNLESS significant movement occured
                jerk_threshold = self.control_panel.get("jerk_threshold")
                if (self.face and self.has_moved(self.face, (x,y,w,h), jerk_threshold*w)) or self.face == None:
                    self.ROI = self.calculate_ROI_around_face(x,y,w,h)
                    self.face=(x,y,w,h)
                
                
                face_found = True
                break
        
        
        if not face_found:  
            # If no face was found, expand the ROI - the subject may have moved too fast    
            self.expand_ROI()

            self.consec_facelss_frames += 1 

            if self.consec_facelss_frames > self.control_panel.get("max_faceless_frames"):
                self.face = None
                self.img_size = 1.0

        if self.face:
            return self.face, face_found
        else:
            return None, False




          



          
                
    



