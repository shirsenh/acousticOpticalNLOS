import numpy as np, matplotlib.pyplot as plt, pickle
import scipy.signal
import sounddevice as sd

v = 340
T = 0.5
fs = 48000
f1 = 20000
f0 = 2000
channels = [1, 1] #speaker, microphone
samples_per_chirp = int(fs*T/channels[0])
t = np.linspace(0, T, int(fs))

chirp = scipy.signal.chirp(t[:samples_per_chirp], f0, T, f1)

chirp_delay_adjusted = np.zeros((2500 + chirp.shape[0]))
chirp_delay_adjusted[:chirp.shape[0]] = chirp

data = {}
raw_data = []

horizontal_pos = 1
vertical_pos = 1

recording_start = 2400
recording_end = recording_start + chirp.shape[0]

for i in range(vertical_pos):
	for j in range(horizontal_pos):
		input('Press if ready for hpos: {} and vpos:{}'.format(j, i))
		myrecording = sd.playrec(chirp_delay_adjusted, fs, channels=2)
		raw_data.append(myrecording[recording_start:recording_end][..., 0])

input('Press to save recordings')
raw_data_reshape = np.zeros((horizontal_pos, 1,  raw_data[0].shape[0], vertical_pos)).astype(np.float64)

index = 0

for i in range(horizontal_pos):
	for j in range(vertical_pos):
		raw_data_reshape[i, 0, :, j] = list(raw_data[index])
		index += 1

# print(raw_data_reshape)
data['raw_scan'] = raw_data_reshape
data['T'] = T
data['f0'] = f0
data['f1'] = f1

file = open('AcousticNLOS/data/distance_far.pkl', 'wb')
pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)