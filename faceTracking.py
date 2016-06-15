import cv2
import numpy as np
import time
from scipy.misc import imresize, imrotate

# Initialize Cascade Calssifiers
faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")

t = time.time()

vid_name = "people.mp4"
ROI_padding = .2
ROI_speed = .10
consec_facelss_frames = 0
max_faceless_frames = 5
jerk_threshold = .03
face_y_cut = .3; face_x_cut = .35
eye_n = 1; eye_scaling = 1.3 
face_n = 1; face_scaling = 1.1
img_size = 1.0 #Shrink by this factor 


# Initialize Web Cam Thread
vs = cv2.VideoCapture(0)
#vs = cv2.VideoCapture("videos/me/" + vid_name)


# Face pixels to be saved for processing
pixels = []

# Read one frame to initialize ROI dimensions
ret, frame = vs.read()

print frame.shape

ROI = ((0, 0), (frame.shape[1], frame.shape[0]))
prev_face = None

while(True):
    # Capture frame
    ret, frame_original = vs.read()
    if frame_original is None:
    	break

    frame = frame_original.copy()
    frame = imresize(frame, img_size)

    # Line Plot
    plot = np.ones((200, 400, 3))

    # Cut out region of interest to reduce computational time. 
    (x1, y1), (x2, y2) = ROI
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)


    # Draw ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255,0,0), thickness=2)

    # Detect faces in the gray image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=face_scaling,
        minNeighbors=face_n,
        minSize=(30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    print faces


    # Initially assume no face was found. Now, search through canidate faces
    face_found = False
    for (x, y, w, h) in faces:

    	#Draw rect around candidate face
        #cv2.rectangle(frame, (x1 + x, y1 + y), (x1+x+w, y1+y+h), (0, 255, 255), 4)
        
        # Check for eyes! If no eyes, no face!
        eyes = eyeCascade.detectMultiScale(
            gray[y:y+h, x:x+w],
            scaleFactor=eye_scaling,
            minNeighbors=eye_n,
            minSize=(int(w*.1), int(h*.1)), #Eyes should not be tiny compared to face
        )
        
        # Shift back x and y to original reference frame
        x += x1; y += y1

        # Were Eyes found?
        if len(eyes) > 0: 

            # Reset faceless frame count
            consec_facelss_frames = 0

            # Calculate New Region of interest. Speeds up computation
            # ROI is an enlarged box around the face
            x1_new, y1_new, x2_new, y2_new =(max(0, x - int(ROI_padding*w)),
                                             max(0, y - int(ROI_padding*h)),
                                             min(frame.shape[1], x + int((1+ROI_padding)*w)),
                                             min(frame.shape[0], y + int((1+ROI_padding)*h)))
        
            # If there was a face last frame, just use that position again
            # UNLESS significant movement occured
            if prev_face:
                (x_old, y_old, w_old, h_old) = prev_face
                
                # Check if new face shifted significantly from the old
                xdiff = abs((x_old + w + x_old)/2. -  (x2_new + x1_new)/2.)
                ydiff = abs((y_old + h + y_old)/2. -  (y2_new + y1_new)/2.)

                # If only slight movement, keep face from previous frame
                small_diff = (xdiff < jerk_threshold*w_old) and (ydiff < jerk_threshold*h_old)
                if small_diff:
                    x,y,w,h = prev_face
                    print "Using previous frame face position"

                # If significant movement, update ROI and use new face
                else:
                    print "prev face exists, but cur face has significantly moved"
                    print "prev face: " , prev_face
                    print "cur face: " , (x,y,w,h)
                    print "ROI: " , ROI
                    
                    ROI = ((x1_new, y1_new), (x2_new, y2_new))
                    prev_face=(x,y,w,h)
            
            # No previous face; we MUST use this new face
            else:
                ROI = ((x1_new, y1_new), (x2_new, y2_new))
                prev_face=(x,y,w,h)
                print "No prev face, using new face"
                print "cur face:" , (x,y,w,h)
                print "ROI:" , ROI

            
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

            # I want the face pixels on the original, non-resized image
            y_org = int(y/img_size); x_org = int(x/img_size); h_org = int(h/img_size); w_org = int(w/img_size)       
            
            # Grab middle of face and append mean pixels
            face = frame_original[y_org + int(face_y_cut*h_org): y_org + h_org - int(face_y_cut*h_org), 
                         x_org + int(face_x_cut*w_org): x_org + w_org - int(face_x_cut*w_org)]
            pixels.append([face[:, :,  0].mean(), 
                           face[:, :,  1].mean(), 
                           face[:, :,  2].mean()
                         ])

            cv2.imshow("face", imresize(face, (300,300)))

            # Draw Face rectangle!
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            midx = (x + w + x)/2; midy = (y+h+y)/2
            cv2.circle(frame, (midx,midy), 5, color=(0,0,0)) 

            # Draw rects around eyes 
            for eye in eyes:
                (xe, ye, we, he) = eye      
                cv2.rectangle(frame, (x+xe, y+ye), (xe+x+we, ye+y+he), (0, 255, 0), 1)

                         
            face_found = True
            break
    
    # If no face was found, expand the ROI - may have moved too fast
    if not face_found:
        ROI_w, ROI_h = x2-x1, y2-y1
        ROI =((max(0, x1 - int(ROI_speed*ROI_w)), 
               max(0, y1 - int(ROI_speed*ROI_h))),

              (min(frame.shape[1], x2  + int(ROI_speed*ROI_w)), 
               min(frame.shape[0], y2+int(ROI_speed*ROI_h)))) 

        consec_facelss_frames += 1 

        if consec_facelss_frames > max_faceless_frames:
            prev_face = None      

        if prev_face:
            x,y,w,h = prev_face 
                         
            y_org = int(y/img_size); x_org = int(x/img_size); h_org = int(h/img_size); w_org = int(w/img_size)       
            
            # Grab middle of face and append mean pixels
            face = frame_original[y_org + int(face_y_cut*h_org): y_org + h_org - int(face_y_cut*h_org), 
                         x_org + int(face_x_cut*w): x_org + w_org - int(face_x_cut*w_org)]
            pixels.append([face[:, :,  0].mean(), 
                           face[:, :,  1].mean(), 
                           face[:, :,  2].mean()
                         ])

            #cv2.imshow("face", imresize(face, (300,300)))

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            midx = (x + w + x)/2; midy = (y+h+y)/2
            cv2.circle(frame, (midx,midy), 5, color=(0,0,0))
    
    

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "%.2f"%(1 / (time.time() - t)),(0,30), font, 1, (0,0,0),2)
    
    # Display the resulting frame
    cv2.imshow('main', frame)  

    t = time.time()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


np.savetxt("data/" + vid_name.split(".")[0] + ".txt", np.array(pixels))  

print vs.get(cv2.cv.CV_CAP_PROP_FPS)

cv2.destroyAllWindows()
vs.release()

