import numpy as np
import cv2
import matplotlib.pyplot as plt

class FaceStats:
	mean_top_face_pixels = []
	mean_bottom_face_pixels = []
	mean_face_pixels = []
	face = None

	def __init__(self, FPS):
		self.FPS = FPS

	def normalizeRGB(self, x):
		r, g, b = x[:,0], x[:,1], x[:,2]
		r = (r - r.mean()) / r.std()
		g = (g - g.mean()) / g.std()
		b = (b - b.mean()) / b.std()
		return np.array([r, g, b])

	def face_fourier(self, mean_face_pixels):
		signals = []
		x = np.array(mean_face_pixels)
		r,g,b = self.normalizeRGB(x)
		for cs, color in zip(['r','g','b'], [r,g,b]):

		    n = len(color)
		    color *= np.hanning(n)

		    # Fourier transform
		    I = abs(np.fft.fft(color)[:n/2])
		    freqs = np.fft.fftfreq(n, 1./self.FPS)[:n/2]
		    
		    mask = (freqs > .8) & (freqs < 2.5)
		    signals.append((freqs[mask], I[mask]))
		
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


	def draw_signal(self, signal, plot, color=(0,0,0)):
		x_scale = plot.shape[1] / len(signal)
		
		p = signal[0]
		for ix, point in enumerate(signal):
		    cv2.line(plot, ((ix-1)*x_scale, 235 - int(p)),
				(ix*x_scale, 235 - int(point)),color, 1)
		    p = point

	def draw_x_axis(self, x_arr, plot):
		for ix, x in enumerate(x_arr):
			cv2.putText(plot, "%.1f"%x,(ix * plot.shape[1] / len(x_arr),250), 
				        cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0),1)


	def draw_face_fourier(self, width=400, window=15):
		plot = np.ones((255,width,3))
		colors = [(255,0,0), (0,255,0), (0,0,255)]

		face_pixels = self.mean_face_pixels[-window*self.FPS:]
		for i, (freqs, signal) in enumerate(self.face_fourier(face_pixels)):
			signal *= 215. / max(signal)
			
			if i!=1:
				continue

			self.draw_signal(signal, plot, colors[i])


			# Annotate Bottom
			f = np.linspace(.8, 2.5, 6)*60
			self.draw_x_axis(f, plot)

			#	Mark Peak
			x,y = (int(signal.argmax() * width/len(signal)), 
				   235-int(signal.max()))
			cv2.circle(plot,(x,y),5, (0,0,255), -1)
			peak_x = freqs[signal.argmax()] * 60
			cv2.putText(plot, "%d"%peak_x + " BPM", (x,y-5), 
				        cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0),1)


		cv2.imshow("Heart Rate", plot)




	def save_face_pixels(self, path):
		np.savetxt(path, np.array(self.mean_face_pixels))

