import cv2
import numpy as np
from imutils.video import WebcamVideoStream
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time
from scipy.misc import imresize

# Initialize Cascade Calssifiers
faceCascade = cv2.CascadeClassifier("/usr/local/Cellar/opencv/2.4.13/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("/usr/local/Cellar/opencv/2.4.13/share/OpenCV/haarcascades/haarcascade_eye.xml")

# Initialize Web Cam Thread
#vs = VideoStream(src="/Users/devinwieker/Desktop/Metis/faceTrack/videos/me/people.mp4").start()
#vs = VideoStream(src="/Users/devinwieker/Desktop/Metis/faceTrack/videos/me/cam.MOV").start()
vs = cv2.VideoCapture("/Users/devinwieker/Desktop/Metis/faceTrack/videos/me/finger.MOV")
fps = FPS().start()


# Face fixels to be saved for processing
pixels = []

# Read one frame to initialize ROI dimensions
ret, frame = vs.read()
ROI = ((0, 0), (frame.shape[1], frame.shape[0]))

while(True):
    # Capture frame
    ret, frame = vs.read()

    if frame is None:
        break

    # Line Plot
    plot = np.ones((200, 400, 3))


    x1, y1 = int(.2*frame.shape[1]), int(frame.shape[0]*.2)
    x2, y2 = int(.7*frame.shape[1]), int(frame.shape[0]*.35)
           
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 4)


    face = frame[y1:y2, x1:x2]
    pixels.append([face[:, :,  0].mean(), 
                   face[:, :,  1].mean(), 
                   face[:, :,  2].mean()
                  ])


    # # Display line plot
    # if len(pixels) / fps.fps() > 20:
    #     pixels.pop(0)

    #     numpy_pixels = np.array(pixels)
    #     r, g, b = numpy_pixels[:,0], numpy_pixels[:,1], numpy_pixels[:,2]
    #     r = (r - r.mean()) / r.std()
    #     g = (g - g.mean()) / g.std()
    #     b = (b - b.mean()) / b.std()
    #     p = (r[0], g[0], b[0])
    #     x_scale = float(plot.shape[1]) / len(g)

    #     print x_scale

    #     y_offset =  plot.shape[0] / 2
    #     for i, point in enumerate(zip(r,g,b)):
    #         cv2.line(plot, ( int((i-1)*x_scale), y_offset + int(p[0]*30)), 
    #                  (int(i*x_scale), y_offset + int(point[0]*30)), 
    #                  (255, 0, 0), 2)

    #         cv2.line(plot, (int((i-1)*x_scale), y_offset + int(p[1]*30)), 
    #                  (int(i*x_scale), y_offset + int(point[1]*30)), 
    #                  (0, 255, 0), 2)

    #         cv2.line(plot, (int((i-1)*x_scale), y_offset + int(p[2]*30)), 
    #                  (int(i*x_scale), y_offset + int(point[2]*30)), 
    #                  (0, 0, 255), 2)
            

    #         p = point

    #     # Display the resulting frame
    #     cv2.imshow('plot', plot) 



    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, str(fps.fps()),(0,30), font, 1, (255,0,255),2)

    # Display the resulting frame
    cv2.imshow('main', frame)  

    fps.update()
            
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# do a bit of cleanup
fps.stop()    

#np.savetxt("finger.txt", np.array(pixels))

print fps.fps()
print vs.get(cv2.cv.CV_CAP_PROP_FPS)

cv2.destroyAllWindows()
vs.stop()



