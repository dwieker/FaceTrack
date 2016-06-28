import cv2
import numpy as np
import time
from scipy.misc import imresize, imrotate
from faceTracker import FaceTracker
from stats import FaceStats
import sys
from ControlPanel import ControlPanel

t = time.time()
face_slice = .20, .80, .0, .8

vid_name = "people.mp4"
vs = cv2.VideoCapture("videos/" + vid_name)

FPS = vs.get(cv2.cv.CV_CAP_PROP_FPS)
if not FPS:
    print "Unknown FPS!"
    vs.release()
    sys.exit()
FPS = int(FPS)


control_panel = ControlPanel()
faceTracker = FaceTracker(control_panel)
faceStats = FaceStats(FPS, control_panel)

fnum = 0
while(True):
    
    # Capture frame
    ret, frame = vs.read()
    if frame is None:
    	break
   
    faceTracker.update_frame(frame)

    face_coords, face_found = faceTracker.locate_face()
    if face_coords:
        x,y,w,h = face_coords
      
        ls, rs, ts, bs = face_slice
        x2 = x + int(rs*w); y2 = y + int(bs*h) 
        face = frame[y+int(ts*h):y2, x+int(ls*w):x2]
        faceStats.update_face(face)

        # Draw Face rectangle!
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255*(not face_found)), 2)
        
        # Draw circle at center of face!
        midx = (x + w + x)/2; midy = (y+h+y)/2
        cv2.circle(frame, (midx,midy), 3, color=(0,0,0), thickness=-1) 

        # window = control_panel.get("window")
        # if not np.isnan(faceStats.mean_face_pixels[:window*FPS]).any():
        #     faceStats.draw_ICA()
        #     faceStats.draw_normalized_signal()

    time_elapsed = time.time() - t
    adjusted_fps = FPS*control_panel.get("FPS_scaling")
    if time_elapsed < (1. / adjusted_fps):
        time.sleep(1./adjusted_fps - time_elapsed)

    # Draw FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "%.2fFPS"%(1 / (time.time() - t)),(0,30), font, 1, (0,0,0),2)

    total_t = float(fnum) / FPS
    cv2.putText(frame, "%.2f"%total_t+"s",(0,60), font, 1, (0,0,0),2)


    # Display the frame
    display_scaling = control_panel.get("display_scaling")
    cv2.imshow('main', imresize(frame, display_scaling))  

    t = time.time()
    fnum += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#faceStats.save_face_pixels("data/" + vid_name.split(".")[0] + ".txt")  
vs.release()
cv2.destroyAllWindows()


