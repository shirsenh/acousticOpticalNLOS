import numpy as np, matplotlib.pyplot as plt, pickle
import scipy.signal, os
import sounddevice as sd

v = 340
channels = [14, 14]
T = 0.125 * channels[0]
fs = 48000
f1 = 20000
f0 = 2000
samples_per_chirp = int(fs*T / channels[0])
t = np.linspace(0, T, int(fs * T))

chirp = scipy.signal.chirp(t[:samples_per_chirp], f0, T/channels[1], f1)
chirp_delay_adjusted = np.zeros((2500 + chirp.shape[0]))
chirp_delay_adjusted[:chirp.shape[0]] = chirp

data = {}
raw_data = []

horizontal_pos = 32
vertical_pos = channels[0]

recording_start = 2400
recording_end = recording_start + chirp.shape[0]

for i in range(vertical_pos):
	for j in range(horizontal_pos):
		input('Press if ready for hpos: {} and vpos:{}'.format(j, i))
		myrecording = sd.playrec(chirp_delay_adjusted, fs, channels=1)
		raw_data.append(myrecording[recording_start:recording_end][..., 0])

input('Press to save recordings')
# vertical positions = channels
raw_data_reshape = np.zeros((horizontal_pos, 1, raw_data[0].shape[0]*vertical_pos, vertical_pos))

index = 0
mult_mat = np.array([
	[1, np.random.uniform(0.95, 0.82), np.random.uniform(0.81, 0.68), np.random.uniform(0.67, 0.50), np.random.uniform(0.49, 0.32), 0.65, 0.58, 0.52, 0.48, 0.44, 0.40, 0.38], 
	[np.random.uniform(0.95, 0.82), 1, np.random.uniform(0.95, 0.82), np.random.uniform(0.81, 0.68), np.random.uniform(0.67, 0.50), 0.70, 0.65, 0.58, 0.52, 0.48, 0.44, 0.40], 
	[np.random.uniform(0.81, 0.68), np.random.uniform(0.95, 0.82), 1, np.random.uniform(0.95, 0.82), np.random.uniform(0.81, 0.68), 0.76, 0.70, 0.65, 0.58, 0.52, 0.48, 0.44], 
	[np.random.uniform(0.67, 0.50), np.random.uniform(0.81, 0.68), np.random.uniform(0.95, 0.82), 1, np.random.uniform(0.95, 0.82), 0.85, 0.76, 0.70, 0.65, 0.58, 0.52, 0.48], 
	[np.random.uniform(0.49, 0.32), np.random.uniform(0.67, 0.50), np.random.uniform(0.81, 0.68), np.random.uniform(0.95, 0.82), 1, 0.92, 0.85, 0.76, 0.70, 0.65, 0.58, 0.52], 
	[0.65, 0.70, 0.76, 0.85, 0.92, 1, 0.92, 0.85, 0.76, 0.70, 0.65, 0.58], 
	[0.58, 0.62, 0.70, 0.76, 0.85, 0.92, 1, 0.92, 0.85, 0.76, 0.70, 0.65], 
	[0.52, 0.58, 0.62, 0.70, 0.76, 0.85, 0.92, 1, 0.92, 0.85, 0.76, 0.70], 
	[0.48, 0.52, 0.58, 0.62, 0.70, 0.76, 0.85, 0.92, 1, 0.92, 0.85, 0.76], 
	[0.44, 0.48, 0.52, 0.58, 0.62, 0.70, 0.76, 0.85, 0.92, 1, 0.92, 0.85], 
	[0.40, 0.44, 0.48, 0.52, 0.58, 0.62, 0.70, 0.76, 0.85, 0.92, 1, 0.92], 
	[0.38, 0.40, 0.44, 0.48, 0.52, 0.58, 0.62, 0.70, 0.76, 0.85, 0.92, 1]]) 

for j in range(vertical_pos):
	for i in range(horizontal_pos):
		raw_data_reshape[i, 0, j*samples_per_chirp:(j + 1)*samples_per_chirp, j] = list(raw_data[index])
		index += 1

# for i in range(horizontal_pos):
# 	for j in range(vertical_pos):
# 		for k in range(vertical_pos):
# 			raw_data_reshape[i, 0, k*samples_per_chirp:(k+1)*samples_per_chirp, j] = list(raw_data[index]*mult_mat[j, k])
# 	index += 1



data['raw_scan'] = raw_data_reshape
data['T'] = T
data['f0'] = f0
data['f1'] = f1

outdir = 'AcousticNLOS/data_new/'

if not os.path.exists(outdir):
	os.makedirs(outdir)

file = open(os.path.join(outdir, '2bottles.pkl'), 'wb')
pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

# Box 1 is taken along the length of the room
