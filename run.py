import cv2
import numpy as np
import time
from scipy.misc import imresize, imrotate
from faceTracker import FaceTracker
from stats import Stats


t = time.time()
face_slice = .20, .80, .0, .8

vid_name = "dad.mov"

# Initialize Video Thread
#vs = cv2.VideoCapture(0)
vs = cv2.VideoCapture("videos/" + vid_name)

# Face pixels to be saved for processing
pixels = []



faceTracker = FaceTracker(eye_n=0, eye_scaling=1.1, eye_jerk_threshold=0.1, img_size=.50)
faceStats = Stats()

fnum = 0
while(True):
    
    # Capture frame
    ret, frame = vs.read()
    if frame is None:
    	break
   
    faceTracker.update_frame(frame)

    face_coords, face_found = faceTracker.locate_face()
    if face_found:
        
        x,y,w,h = face_coords
       
      
        ls, rs, ts, bs = face_slice
        x2 = x + int(rs*w); y2 = y + int(bs*h) 
        face = frame[y+int(ts*h):y2, x+int(ls*w):x2]
        faceStats.update_face(face)

        # Display the face
        cv2.imshow("face", face)


        # Draw Face rectangle!
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255*(not face_found)), 2)
        
        # Draw circle at center of face!
        midx = (x + w + x)/2; midy = (y+h+y)/2
        cv2.circle(frame, (midx,midy), 5, color=(0,0,0)) 

        
        eye_band = faceTracker.get_eye_band()
        if eye_band:
            xi, yi, wi, hi = eye_band
            cv2.rectangle(frame, (xi,yi), (xi+wi, yi+hi) , color=(0,255,255*(not face_found)))

            faceStats.update_eyes(frame[yi:yi+h, xi:xi+w])

            # Display line plot
            #faceStats.draw_eye_mean()

        if fnum > 500:
            faceStats.draw_face_fourier()

    # Draw ROI
    #(x1, y1), (x2, y2) = faceTracker.get_ROI()
    #cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255,0,0), thickness=2)

    # Draw FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "%.2f"%(1 / (time.time() - t)),(0,30), font, 1, (0,0,0),2)
    
    # Display the resulting frame
    cv2.imshow('main', frame)  


    t = time.time()
    fnum += 1


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


faceStats.save_face_pixels("data/" + vid_name.split(".")[0] + ".txt")  

cv2.destroyAllWindows()
vs.release()

