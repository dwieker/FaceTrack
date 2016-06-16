import cv2
import numpy as np
from scipy.misc import imresize, imrotate

class FaceTracker():

    # Initialize Cascade Calssifiers
    faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    eyeCascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")

    def __init__(   self,
                    ROI_padding = .2,
                    ROI_speed = .10,
                    consec_facelss_frames = 0,
                    max_faceless_frames = 5,
                    jerk_threshold = .03,
                    eye_jerk_threshold = .05,
                    eye_n = 1, eye_scaling = 1.2,
                    face_n = 1, face_scaling = 1.1,
                    img_size = 1.0,
                    eye_span = .85):
    

        self.ROI_padding = ROI_padding 
        self.ROI_speed = ROI_speed 
        self.consec_facelss_frames = consec_facelss_frames
        self.max_faceless_frames = max_faceless_frames 
        self.jerk_threshold = jerk_threshold
        self.eye_jerk_threshold = eye_jerk_threshold
        self.eye_n = eye_n
        self.eye_scaling = eye_scaling 
        self.face_n = face_n; self.face_scaling = face_scaling
        self.img_size = img_size
        self.eye_span = eye_span

        self.ROI = None
        self.face = None
        self.frame = None
        self.scaled_gray = None
        self.eyes = []

        # (Height of eyes as a percent of face frame height, and thickness)
        self.eye_band = None 

    def update_frame(self, frame):
        frame = frame.copy()
        frame = imresize(frame, self.img_size)
        
        if self.ROI is None:
            print "Setting ROI to full frame"
            self.ROI = ((0, 0), (frame.shape[1], frame.shape[0]))

        self.frame = frame

    def calculate_ROI_around_face(self, x,y,w,h):
        # Input parameters are a FACE. Output is an exanded ROI, accounting for frame boundaries
        x1_new, y1_new, x2_new, y2_new =(max(0, x - int(self.ROI_padding*w)),
                                        max(0, y - int(self.ROI_padding*h)),
                                        min(self.frame.shape[1], x + int((1+self.ROI_padding)*(w/self.img_size))),
                                        min(self.frame.shape[0], y + int((1+self.ROI_padding)*(h/self.img_size)))) 
        return ((x1_new, y1_new), (x2_new, y2_new))

    def expand_ROI(self):
        (x1, y1), (x2, y2) = self.ROI
        ROI_w, ROI_h = x2-x1, y2-y1
        self.ROI =((max(0, x1 - int(self.ROI_speed*ROI_w)), 
                    max(0, y1 - int(self.ROI_speed*ROI_h))),
                   (min(self.frame.shape[1], x2  + int(self.ROI_speed*ROI_w)), 
                    min(self.frame.shape[0], y2+int(self.ROI_speed*ROI_h)))) 


    def has_moved(self, old, new, threshold):
        (x_old, y_old, w_old, h_old) = old
        (x,y,w,h) = new
        xdiff = abs((x_old + w_old + x_old)/2. -  (x + w + x)/2.)
        ydiff = abs((y_old + h_old + y_old)/2. -  (y + h + y)/2.)
        print xdiff, ydiff
        return (xdiff > threshold) or (ydiff > threshold)


    def get_eye_band(self):
        # Returns a rectangle x,y,w,h that encompasses the eyes
        # Must locate face first!
        if self.face != None:
            x,y,w,h = self.face
            if self.eye_band == None:
                #  We must accurately locate the eyes! #Attempt to set it
                eyes = FaceTracker.eyeCascade.detectMultiScale(
                    self.scaled_gray[y:y+int(.8*h), x:x+w],
                    scaleFactor=1.1,
                    minNeighbors=2,
                    minSize=(int(w*.1), int(h*.1)), #Eyes should not be tiny compared to face
                )

                if len(eyes) > 0:
                    xi,yi,hi,wi = eyes[0]
                    self.eye_band = yi / float(h), hi / float(h)
            
            if self.eye_band:
                hi, thickness = self.eye_band
                return (int((x+w*(1-self.eye_span)) / self.img_size), int((y+h*hi)/self.img_size), 
                       int(w*(2*self.eye_span-1)/self.img_size), int((thickness*h)/self.img_size))


    def locate_face(self):
    
        # Cut out region of interest to reduce computational time. 
        (x1, y1), (x2, y2) = self.ROI
        gray = cv2.cvtColor(self.frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        self.scaled_gray = gray

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
      
                x += x1; y += y1;

                # Reset faceless frame count
                self.consec_facelss_frames = 0
            
                # If there was a face last frame, just use that position again
                # UNLESS significant movement occured
                if (self.face and self.has_moved(self.face, (x,y,w,h), self.jerk_threshold*w)) or self.face == None:
                    self.ROI = self.calculate_ROI_around_face(x,y,w,h)
                    self.face=(x,y,w,h)
                
                
                face_found = True
                break
        
        
        if not face_found:  
            # If no face was found, expand the ROI - the subject may have moved too fast
        
            print "Face not found..."
            
            self.expand_ROI()

            self.consec_facelss_frames += 1 

            if self.consec_facelss_frames > self.max_faceless_frames:
                self.face = None
                print "Face set to None"


        if self.face:

            # Transform the face coordinates to the original images reference frame.
            x,y,w,h = self.face
            y = int(y/self.img_size); x = int(x/self.img_size); 
            h = int(h/self.img_size); w = int(w/self.img_size) 

            return (x,y,w,h), face_found


        return None, False




          



          
                
    



