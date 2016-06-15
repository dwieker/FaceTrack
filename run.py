import cv2
import numpy as np
import time
from scipy.misc import imresize, imrotate
from faceTracker import FaceTracker


t = time.time()

vid_name = "me4.mov"

# Initialize Video Thread
#vs = cv2.VideoCapture(0)
vs = cv2.VideoCapture("videos/" + vid_name)

# Face pixels to be saved for processing
pixels = []

faceTracker = FaceTracker(eye_n=2)
while(True):
    
    # Capture frame
    ret, frame = vs.read()
    if frame is None:
    	break
   
    faceTracker.update_frame(frame)

    face_coords = faceTracker.locate_face()
    if face_coords:
        
        x,y,w,h = face_coords
        face = frame[y:y+h, x:x+w]

        pixels.append([face[:, :,  0].mean(), 
                       face[:, :,  1].mean(), 
                       face[:, :,  2].mean()
                     ])

     
        # Draw Face rectangle!
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw circle at center of face!
        midx = (x + w + x)/2; midy = (y+h+y)/2
        cv2.circle(frame, (midx,midy), 5, color=(0,0,0)) 


        # Draw eyes!
        for (xi, yi, wi, hi) in faceTracker.locate_eyes():
            cv2.rectangle(frame, (xi,yi), (xi+wi, yi+hi) , color=(0,255,0)) 


    # # Display line plot
    # if len(pixels) >= 30:
    #     numpy_pixels = np.array(pixels[-30:])
    #     r, g, b = numpy_pixels[:,0], numpy_pixels[:,1], numpy_pixels[:,2]
    #     r = (r - r.mean()) / r.std()
    #     g = (g - g.mean()) / g.std()
    #     b = (b - b.mean()) / b.std()
    #     p = (r[0], g[0], b[0])
    #     x_scale = 400 / len(g)
    #     y_offset =  100
    #     for i, point in enumerate(zip(r,g,b)):
    #         cv2.line(plot, ((i-1)*x_scale, y_offset + int(p[0]*30)), 
    #                  (i*x_scale, y_offset + int(point[0]*30)), 
    #                  (255, 0, 0), 2)

    #         cv2.line(plot, ((i-1)*x_scale, y_offset + int(p[1]*30)), 
    #                  (i*x_scale, y_offset + int(point[1]*30)), 
    #                  (0, 255, 0), 2)

    #         cv2.line(plot, ((i-1)*x_scale, y_offset + int(p[2]*30)), 
    #                  (i*x_scale, y_offset + int(point[2]*30)), 
    #                  (0, 0, 255), 2)
            

    #         p = point
    
    #     # Display the resulting frame
    #     cv2.imshow('plot', plot) 

    # Draw ROI
    (x1, y1), (x2, y2) = faceTracker.ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255,0,0), thickness=2)

    # Draw FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "%.2f"%(1 / (time.time() - t)),(0,30), font, 1, (0,0,0),2)
    
    # Display the resulting frame
    cv2.imshow('main', frame)  

    t = time.time()


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#np.savetxt("data/" + vid_name.split(".")[0] + ".txt", np.array(pixels))  

cv2.destroyAllWindows()
vs.release()
