import wave
import scipy
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

WAV_FILES = ['input/%d.wav' % i for i in range(1,6)]
OUT_FILES = ['output/%d.wav' % i for i in range(1,6)]
PLOT_SAMPLE_RATE = 0.01
BIT = 32 - 1

def get_decibels(wav):
    ret = []
    chunks = np.array_split(wav, wav.shape[0] * PLOT_SAMPLE_RATE)
    chunks = list(map(lambda x:x.astype(np.float32), chunks))
    for chunk in chunks:
        rms = np.mean(np.sqrt(np.abs(chunk ** 2)))
        db = 0
        if rms != 0:
            db = math.log2(rms) * 6.0  - 80 # our dB
        ret.append(db)
    return np.array(ret)

wav_path = WAV_FILES[0]
out_path = OUT_FILES[0]

rate, wav_raw = scipy.io.wavfile.read(wav_path)
wav_raw *= 2 ** BIT
wav_out = wav_raw * 0.01 
wav_db = get_decibels(wav_out)

wavfile.write(out_path, rate, wav_out / (2 ** BIT))

#########
plots = [
    scipy.signal.resample(wav_raw, int(wav_raw.shape[0] * PLOT_SAMPLE_RATE)),
    scipy.signal.resample(wav_out, int(wav_out.shape[0] * PLOT_SAMPLE_RATE)),
    scipy.signal.resample(wav_db, 100),
    ]

fig = plt.figure()

ax1 = fig.add_subplot(221)
ax1.set_title(wav_path)
ax1.plot(plots[0])

ax2 = fig.add_subplot(222)
ax2.set_title(out_path)
ax2.plot(plots[1])

ax3 = fig.add_subplot(223)
ax3.set_title('input db')
ax3.plot(plots[2])


plt.show()