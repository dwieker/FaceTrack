import cv2
import numpy as np
import time
from scipy.misc import imresize, imrotate
from faceTracker import FaceTracker
from stats import FaceStats


t = time.time()
face_slice = .20, .80, .0, .8

vid_name = "dad.mov"
FPS = 30

# Initialize Video Thread
#vs = cv2.VideoCapture(0)
vs = cv2.VideoCapture("videos/" + vid_name)

# Face pixels to be saved for processing
pixels = []



faceTracker = FaceTracker(eye_n=0,
                          jerk_threshold=.10, 
                          img_size=.3)
faceStats = FaceStats(FPS=FPS)

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

        cv2.imshow("face", face)

        # Draw Face rectangle!
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255*(not face_found)), 2)
        
        # Draw circle at center of face!
        midx = (x + w + x)/2; midy = (y+h+y)/2
        cv2.circle(frame, (midx,midy), 5, color=(0,0,0)) 

        if fnum > 5*FPS:
            faceStats.draw_face_fourier(window=15)


            # # Display the face, but with color AMPLIFIED!
            # green = [c[1] for c in faceStats.mean_face_pixels[-5*FPS:]]
            # avg = sum(green) / len(green)

            # print avg, green[-1]
            # shift = int(3000 * (avg - green[-1]) / avg)

            # if shift > 50:
            #     shift = 50
            # if shift < -50:
            #     shift = -50
           
            # shift += 50

            # print shift

            # face_copy = face.copy() #.astype(np.int16)
            # face_copy[:,:,1] += shift
            # cv2.imshow("face", face_copy)


    # Draw FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "%.2fFPS"%(1 / (time.time() - t)),(0,30), font, 1, (0,0,0),2)
    cv2.putText(frame, "%.2f"%(float(fnum) / FPS)+"s",(int(frame.shape[1]*.7),30), font, 1, (0,0,0),2)


    # Display the resulting frame
    cv2.imshow('main', frame)  


    t = time.time()
    fnum += 1


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


faceStats.save_face_pixels("data/" + vid_name.split(".")[0] + ".txt")  
cv2.destroyAllWindows()
vs.release()

