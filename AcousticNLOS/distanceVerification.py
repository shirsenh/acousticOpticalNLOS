import numpy as np
from numpy.fft import ifftn, fftn
import matplotlib.pyplot as plt
import util.lct as lct
import pickle
from tqdm import tqdm
from scipy.signal import firwin, lfilter
import scipy.signal
import csv
import os
from util.pickle_util import *
import sys
import time
plt.style.use('dark_background')

class distanceVerification:
    """docstring for distanceVerification"""
    def __init__(self):
        self.enable_plotting = 1
        self.T = 0.125
        self.v = 340
        self.fs = 48000
        self.f0 = 2000
        self.f1 = 20000
        self.channels = [5, 5]
        self.max_dist = int(self.T*self.v/2 / self.channels[0])
        self.samples_per_chirp = int(self.fs*self.T)
        self.B = self.f1 - self.f0
        self.f = np.linspace(0, self.fs, self.samples_per_chirp)
        self.t = np.linspace(0, self.T, int(self.fs*self.T))
        self.calibration = self.getCalibrationArray()
        self.chirp = self.getChirpSignal()
        self.hp, self.lp = self.getFilterParameters(1.5e3, self.B)

    def getCalibrationArray(self): 
        mic_ids = [1000]
        calibration_freq = np.zeros((256, self.channels[0]))
        calibration_val = np.zeros((256, self.channels[0]))

        for idx, mic_id in enumerate(mic_ids):
            if not mic_id:
                continue
            with open('calibration_files/' + str(mic_id) + '.txt', 'r') as fid:
                reader = csv.reader(fid, dialect="excel-tab")
                reader = list(reader)[2:]
                calibration_freq[:, idx] = np.array([np.float32(reader[i][0]) for i in range(256)])
                calibration_val[:, idx] = np.array([np.float32(reader[i][1])/2000 for i in range(256)])

        # interpolate to our frequencies
        fx = np.linspace(-self.fs/2, self.fs/2, self.samples_per_chirp)
        x = np.abs(np.fft.fftshift(fx))
        calibration = np.zeros((self.samples_per_chirp, self.channels[0]))
        for idx in range(self.channels[0]):
            calibration[:, idx] = np.interp(x, calibration_freq[:, idx], calibration_val[:, idx])

        # convert from log space to scale factor
        calibration = 10**(calibration / 20)
        return calibration

    def getFilterParameters(self, fc_lp, fc_hp, n=511):
        # get lowpass and highpass filters
        n = 511
        fc = 1.5e3
        hp = firwin(n, fc_lp, fs=self.fs, pass_zero=False)
        lp = firwin(n, fc_hp, fs=self.fs, pass_zero=True)
        return hp, lp

    def getChirpSignal(self):
        chirp = scipy.signal.chirp(self.t[:int(self.samples_per_chirp)], 
                                       self.f0, self.T/self.channels[1], self.f1)
        return chirp

    def demodulateFMCW(self, input_file, output_file, overwrite_raw=False):

        if not os.path.isfile(output_file) or overwrite_raw:
            ckpt = pickle_load(input_file)
            raw_scan = ckpt['raw_scan']
            # print(np.sum(raw_scan))
            
            num_horz_samples = raw_scan.shape[0]
            raw_data = np.zeros((self.channels[0], self.channels[1], num_horz_samples, self.samples_per_chirp))
            print(raw_data.shape)
            for i in range(self.channels[0]):
                for j in range(self.channels[1]):
                    for k in range(num_horz_samples):
                        # print(i, j, k)
                        # raw_data[i, j, k, :] = np.mean(raw_scan[k, :, j*self.samples_per_chirp:(j+1)*self.samples_per_chirp, i], axis=0)
                        raw_data[i, i, k, :] = raw_scan[k, i, :, i]
                        # print(np.mean(raw_scan[k, :, j*self.samples_per_chirp:(j+1)*self.samples_per_chirp, i], axis=0)[0:5], raw_scan[k, :, j*self.samples_per_chirp:(j+1)*self.samples_per_chirp, i][:, 0:5])
                        # sys.exit(-1)

            confocal_data = np.zeros((self.channels[0], num_horz_samples, self.samples_per_chirp))
            for i in range(self.channels[0]):
                for j in range(num_horz_samples):
                    confocal_data[i, j, :] = raw_data[i, i, j, :]

            processed_data = np.zeros((self.channels[0], self.channels[1], num_horz_samples, self.samples_per_chirp))
            # print("hereby")
            for i in tqdm(range(self.channels[0])):
                for j in range(self.channels[1]):
                    for k in range(num_horz_samples):

                        # high pass filter the input data
                        data = lfilter(self.hp, 1, raw_data[i, j, k, :], axis=0) 

                        # take the fourier transform and apply calibration 
                        data_ft = np.fft.fft(data, axis=0)
                        data_ft_cal = data_ft / self.calibration[:, i]
                        data = np.fft.ifft(data_ft_cal, axis=0)

                        # mix with chirp signal
                        data *= self.chirp        

                        # low pass filter the output
                        data = lfilter(self.lp, 1, data, axis=0)

                        # take the fourier transform and display
                        data_ft = np.fft.fft(data, axis=0)
                        data_ft = np.abs(data_ft)**2

                        processed_data[i, j, k, :] = data_ft 

            # save this intermediary output so we don't have to calculate it every time
            np.save(output_file, processed_data)


verif = distanceVerification()
scene = 'box'
fname = os.path.join('data', scene)
verif.fstart = 100
verif.fend = 3000
verif.fstart_idx = np.argmin(abs(verif.f - verif.fstart))
verif.fend_idx = np.argmin(abs(verif.f - verif.fend))
verif.t = np.linspace(verif.fstart/verif.B * verif.max_dist * 2 / verif.v, verif.fend/verif.B * verif.max_dist * 2 / verif.v, verif.fend_idx - verif.fstart_idx)
print('=> Checking scene {}'.format(scene))
print('=> Volume distance {:.02f} m to {:.02f} m'.format(verif.fstart/verif.B * verif.max_dist, verif.fend/verif.B * verif.max_dist))
print('=> Raw data processing')
verif.demodulateFMCW(fname + '.pkl', fname + '.npy', overwrite_raw=True)