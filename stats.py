import numpy as np
import cv2
import matplotlib.pyplot as plt

class Stats:
	mean_face_pixels = []
	mean_eye_pixels = []
	face = None
	eyes = None

	def __init__(self):
		pass

	def normalizeRGB(self, x):
		r, g, b = x[:,0], x[:,1], x[:,2]
		r = (r - r.mean()) / r.std()
		g = (g - g.mean()) / g.std()
		b = (b - b.mean()) / b.std()
		return np.array([r, g, b])

	def face_fourier(self):
		signals = []
		x = np.array(self.mean_face_pixels)
		r,g,b = self.normalizeRGB(x)
		for cs, color in zip(['r','g','b'], [r,g,b]):
		    #Smooth the signal
		#     window_size = 5
		#     window = np.ones(int(window_size))/float(window_size)
		#     color = np.convolve(color, window, 'same')
		    
		    n = len(color)
		    
		    # Fourier transform
		    I = abs(np.fft.fft(color)[:n/2])
		    freqs = np.fft.fftfreq(n, 1./30.)[:n/2]
		    
		    
		    #Filter
		    I[(freqs<.8) | (freqs>2.5)] = 0

		    signals.append(I)
		
		return signals

	def rgb_mean(self, pixels):
		return (
				pixels[:,:,0].mean(),
				pixels[:,:,1].mean(),
				pixels[:,:,2].mean()
			   )

	def rgb2gray(self, rgb):
		r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
		gray = 0.2989 * r + 0.5870 * g + 0.1140 * b 
		return gray


	def update_face(self, face):
		self.face = face
		self.mean_face_pixels.append( self.rgb_mean(face) )

	def update_eyes(self, eye_band):
		self.eyes = self.rgb2gray(eye_band)
		self.mean_eye_pixels.append( self.eyes.mean() )

	def draw_eye_mean(self):
		pixels = self.mean_eye_pixels
		plot = np.ones((255,300,3))
		if len(pixels) >= 30:
			gray = np.array(self.mean_eye_pixels)
			x_scale = 300 / len(gray)
			p = gray[0]
			for i, point in enumerate(gray):
			    cv2.line(plot, ((i-1)*x_scale, int(p)),
					(i*x_scale, int(point)),(0, 0, 255), 2)
			    p = point
	    

	        # Display the resulting frame
	        cv2.imshow('plot', plot) 

	def draw_face_fourier(self):
		plot = np.ones((255,300,3))
		colors = [(255,0,0), (0,255,0), (0,0,255)]
		for i, signal in enumerate(self.face_fourier()):
			x_scale = 300 / len(signal)
			p = signal[0]

			plt.plot(signal)
			plt.show()
			break

			print signal.shape
			for ix, point in enumerate(signal):
			    cv2.line(plot, ((ix-1)*x_scale, int(p)*5),
					(ix*x_scale, int(point)*5),colors[i], 1)
			    p = point

		cv2.imshow("plot", plot)


	def save_face_pixels(self, path):
		np.savetxt(path, np.array(self.face_pixels))

