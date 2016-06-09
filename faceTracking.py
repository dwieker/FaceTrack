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


vid_name = "me"
ROI_scale = .3

# Initialize Web Cam Thread
#vs = VideoStream(src="/Users/devinwieker/Desktop/Metis/faceTrack/videos/me/" + vid_name + ".mov").start()
#vs = cv2.VideoCapture(0)
vs = cv2.VideoCapture("/Users/devinwieker/Desktop/Metis/faceTrack/videos/me/" + vid_name + ".mp4")
#vs = cv2.VideoCapture("/Users/devinwieker/Desktop/Metis/faceTrack/videos/me/people.mp4")
fps = FPS().start()


# Face fixels to be saved for processing
pixels = []

# Read one frame to initialize ROI dimensions
ret, frame = vs.read()

print frame.shape


ROI = ((0, 0), (frame.shape[1], frame.shape[0]))

while(True):
    # Capture frame
    ret, frame = vs.read()

    if frame is None:
    	break

    #frame = imresize(frame, .2)

    # Line Plot
    plot = np.ones((200, 400, 3))

    # Face tracking works on gray scale images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # Region of interest. 
    (x1, y1), (x2, y2) = ROI
    gray = gray[y1: y2, x1 : x2]


    # Equalize histogram
    #gray = cv2.equalizeHist(gray)
    
    # # Draw ROI rectangle
    # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # # Detect faces in the gray image
    # faces = faceCascade.detectMultiScale(
    #     gray,
    #     scaleFactor=2.0,
    #     minNeighbors=2,
    #     minSize=(30, 30),
    #     flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    # )


    # # Initially assume no face was found. Now, search through canidate faces
    # face_found = False
    # for (x, y, w, h) in faces:

    # 	#Draw rect around candidate face
    #     cv2.rectangle(frame, (x1 + x, y1 + y), (x1+x+w, y1+y+h), (0, 255, 255), 4)
        
    #     # Check for eyes! If no eyes, no face!
    #     eyes = eyeCascade.detectMultiScale(
    #         gray[y:y+h, x:x+w],
    #         minNeighbors=1,
    #         minSize=(int(w*.1), int(h*.1)), #Eyes should not be tiny compared to face
    #     )
        
    #     # Were Eyes found?
    #     if len(eyes) > 0: 
            
    #         for eye in eyes:
    #             (xe, ye, we, he) = eye
    #             cv2.rectangle(frame, (x1+x+xe, y1+y+ye), (x1+xe+x+we, y1+ye+y+he), (0, 255, 0), 2)


    #     	# Draw Face bounding box
    #         cv2.rectangle(frame, (x1+x, y1+y), (x1+x+w, y1+y+h), (0, 255, 0), 2)
                 
           
    #         # Grab middle of face and append mean pixels
    #         face = frame[y1 + y + int(.3*h): y1 + y + h - int(.3*h), x1 + x + int(.2*w): x1 + x + w - int(.2*w)]
    #         pixels.append([face[:, :,  0].mean(), 
    #                        face[:, :,  1].mean(), 
    #                        face[:, :,  2].mean()
    #                      ])

    #         cv2.imshow("face", imresize(face, (300,300)))
            
    #         # Display line plot
    #         if len(pixels) >= 30:
    #             numpy_pixels = np.array(pixels[-30:])
    #             r, g, b = numpy_pixels[:,0], numpy_pixels[:,1], numpy_pixels[:,2]
    #             r = (r - r.mean()) / r.std()
    #             g = (g - g.mean()) / g.std()
    #             b = (b - b.mean()) / b.std()
    #             p = (r[0], g[0], b[0])
    #             x_scale = 400 / len(g)
    #             y_offset =  100
    #             for i, point in enumerate(zip(r,g,b)):
    #                 cv2.line(plot, ((i-1)*x_scale, y_offset + int(p[0]*30)), 
    #                          (i*x_scale, y_offset + int(point[0]*30)), 
    #                          (255, 0, 0), 2)

    #                 cv2.line(plot, ((i-1)*x_scale, y_offset + int(p[1]*30)), 
    #                          (i*x_scale, y_offset + int(point[1]*30)), 
    #                          (0, 255, 0), 2)

    #                 cv2.line(plot, ((i-1)*x_scale, y_offset + int(p[2]*30)), 
    #                          (i*x_scale, y_offset + int(point[2]*30)), 
    #                          (0, 0, 255), 2)
                    

    #                 p = point
            
    #             # Display the resulting frame
    #             cv2.imshow('plot', plot) 

                 
    #         # Update Region of interest. Speeds up computation
    #         x1_new, y1_new, x2_new, y2_new = (max(0, x1 + x - int(ROI_scale*w)),
    #                                          max(0, y1 + y - int(ROI_scale*h)),
    #                                          min(frame.shape[1], x1 + x + int((1+ROI_scale)*w)),
    #                                          min(frame.shape[0], y1 + y + int((1+ROI_scale)*h)))
             
    #         ROI = ((x1_new, y1_new), (x2_new, y2_new))
            
               
    #         face_found = True
    #         break
    
    # # If no face found, expand the ROI to full screen
    # if not face_found:
    #     ROI = ((0, 0), (frame.shape[1], frame.shape[0]))


    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(frame, str(fps.fps()),(0,30), font, 1, (0,0,0),2)

    # # Display the resulting frame
    # cv2.imshow('main', frame)  

    print fps.fps()
    
    fps.update()
            
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# do a bit of cleanup
fps.stop()  

#np.savetxt(vid_name + ".txt", np.array(pixels))  

print fps.fps()
print vs.stream.get(cv2.cv.CV_CAP_PROP_FPS)

cv2.destroyAllWindows()
#vs.stop()
vs.release()

