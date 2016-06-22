import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from scipy.ndimage.interpolation import shift

class FaceStats:
	
	def __init__(self, FPS, control_panel, max_window=30):
		self.control_panel = control_panel
		self.FPS = FPS
		self.mean_face_pixels = np.ones((max_window*FPS, 3)) * np.nan
		self.face = None


	def normalizeRGB(self, x):
		r, g, b = x[:,0], x[:,1], x[:,2]
		r = (r - r.mean()) / r.std()
		g = (g - g.mean()) / g.std()
		b = (b - b.mean()) / b.std()
		return np.array([r, g, b])

	def fourier(self, pixels):
		n = len(pixels)
		freqs = np.fft.fftfreq(n, 1./self.FPS)[:n/2]
		mask = (freqs > .8) & (freqs < 2.5)

		pixels *= np.hanning(n)

		I = abs(np.fft.fft(pixels)[:n/2])
	
		return freqs[mask], I[mask]

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
		self.mean_face_pixels[1:] = self.mean_face_pixels[:-1]
		self.mean_face_pixels[0] = self.rgb_mean(face)


	def draw_signal(self, signal, plot, color=(0,0,0)):
		x_scale = plot.shape[1] / len(signal)
		
		p = signal[0]
		for ix, point in enumerate(signal):
		    cv2.line(plot, ((ix-1)*x_scale, 235 - int(p)),
				(ix*x_scale, 235 - int(point)),color, 1)
		    p = point

	def draw_x_axis(self, x_arr, plot):
		for ix, x in enumerate(x_arr):
			cv2.putText(plot, "%.0f"%x,(ix * plot.shape[1] / len(x_arr),250), 
				        cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0),1)


	def draw_face_fourier(self, width=400):
		plot = np.ones((255,width,3))
	
		window = self.control_panel.get("window")
		r,g,b = self.normalizeRGB(self.mean_face_pixels[:window*self.FPS])

		freqs, I = self.fourier(g)

		I *= 215. / max(I)
		
		self.draw_signal(I, plot, (0,255,0))

		# Annotate Bottom
		f = np.linspace(freqs[0], freqs[-1], 6)*60
		self.draw_x_axis(f, plot)

		# Mark Peak
		x,y = (int(I.argmax() * width/len(I)), 
			   235-int(I.max()))
		cv2.circle(plot,(x,y),5, (0,0,255), -1)
		peak_x = freqs[I.argmax()] * 60
		cv2.putText(plot, "%d"%peak_x + " BPM", (x,y-5), 
			        cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0),1)


		cv2.imshow("Raw FFT of green channel", plot)


	def draw_ICA(self, width=400):
		ica = FastICA(n_components=3, max_iter=30, random_state=1)

		window = self.control_panel.get("window")
		r,g,b = self.normalizeRGB(self.mean_face_pixels[:window*self.FPS])
		comps = ica.fit_transform( np.array([r,g,b]).T )

		plot = np.ones((255,width,3))
		f_plot = plot.copy()

		f_transforms = []
		best_signal = None
		max_peak_diff = 0
		for i in range(comps.shape[1]):
			comp = comps[:, i]

			freqs, I = self.fourier(comp)
			I *= 215. / max(I)
			
			f_transforms.append(I)

			peaks = np.where(np.r_[True, I[1:] > I[:-1]] & np.r_[I[:-1] > I[1:], True])[0]
		   
			peak_heights = I[peaks]
			peak_heights.sort()

			if len(peak_heights) < 2:
				peakB = min(I)
			else:
				peakB = peak_heights[-2]
			peak_diff =  (peak_heights[-1] - peakB)
			if peak_diff > max_peak_diff:
				max_peak_diff = peak_diff
				best_signal = i

        # Annotate Bottom
		f = np.linspace(freqs[0], freqs[-1], 6)*60
		self.draw_x_axis(f, f_plot)
	
		# Mark Peak
		f = f_transforms[best_signal]
		x,y = (int(f.argmax() * width/len(f)),235-int(f.max()))
		cv2.circle(f_plot,(x,y),5, (0,0,255), -1)
		peak_x = freqs[f.argmax()] * 60
		cv2.putText(f_plot, "%d"%peak_x + " BPM", (x,y-5), 
			        cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0),1)

		self.draw_signal(f, f_plot, (255,0,0))

		cv2.imshow("ICA best signal - frequency domain", f_plot)

	def draw_raw_signal(self, width=400):
		plot = np.ones((255, width, 3))
		window = self.control_panel.get("window")
		r,g,b = self.normalizeRGB(self.mean_face_pixels[:window*self.FPS])
		colors = [(255,0,0), (0,255,0), (0,0,255)]
		for i, c in enumerate([r,g,b]):
			c = c*50 + 100
			self.draw_signal(c, plot, colors[i])

		t = np.arange(0, window, 2)
		self.draw_x_axis(t, plot)	
		cv2.imshow("Raw RGB signal", plot)


	def save_face_pixels(self, path):
		np.savetxt(path, np.array(self.mean_face_pixels))

