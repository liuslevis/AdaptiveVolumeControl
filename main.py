import wave
import scipy
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

WAV_FILES = ['./input/%d.wav' % i for i in range(1,4)] + ['./input/3.mid.wav', './input/3.loud.wav']
OUT_FILES = [path.replace('./input/', './output/') for path in WAV_FILES]
PLOT_PATH = 'output/fig.png'

PLOT_WAV_SAMPLE_NUM = 10000
PLOT_WAV_MAX = 0.01
PLOT_DB_SAMPLE_NUM = 100
PLOT_DB_MIN = 0
PLOT_DB_MAX = 100
CHUNK_SIZE = 100
P_ref = 1 / 2 ** 14

def Lp(Prms):
    if Prms != 0:
        return 20 * math.log10(Prms / P_ref)
    return 0

def Prms(Lp):
    return 10 ** (Lp / 20) * P_ref

def db(wav):
    ret = []
    chunks = np.array_split(wav, CHUNK_SIZE)
    chunks = list(map(lambda x:x.astype(np.float32), chunks))
    for chunk in chunks:
        prms = np.mean(np.sqrt(np.abs(chunk ** 2)))
        db = Lp(prms)
        ret.append(db)
    return np.array(ret)

def mean_db(wav):
    return np.mean(db(wav))

def calc_gain(wav):
    n = 20
    i = 0
    gain = 1.0
    target = 55
    epsilon = 2
    while True:
        i += 1
        if i >= n: break

        db = mean_db(wav * gain)

        if abs(db - target) < epsilon: break

        if db > target + 40:
            gain *= 0.0005
        elif db > target + 30:
            gain *= 0.05
        elif db > target + 20:
            gain *= 0.1
        elif db > target + 10:
            gain *= 0.5
        elif db > target + 0:
            gain *= 0.7
        elif db > target - 10:
            gain *= 1.2
        elif db > target - 20:
            gain *= 1.5
        elif db > target - 30:
            gain *= 2.0

        print('\tgain:%.2f db:%.2f -> %.2f' %(gain, db, mean_db(wav * gain)))

    return gain

def auto_gain_control(wav):
    gain = calc_gain(wav)
    ret = wav * gain
    return ret

N = len(WAV_FILES)
wav_paths = WAV_FILES
out_paths = OUT_FILES
rates = [None for i in range(N)]
wav_raws = [None for i in range(N)]
wav_outs = [None for i in range(N)]
wav_out_dbs  = [None for i in range(N)]
wav_raw_dbs  = [None for i in range(N)]

for i in range(N):
    rates[i], wav_raws[i] = scipy.io.wavfile.read(wav_paths[i])
    print('\nauto_gain_control:', wav_paths[i])
    wav_outs[i] = auto_gain_control(wav_raws[i])
    wav_out_dbs[i] = db(wav_outs[i])
    wav_raw_dbs[i] = db(wav_raws[i])

    scipy.io.wavfile.write(out_paths[i], rates[i], wav_outs[i])

#########
print('\nplotting:', PLOT_PATH)
plots = []

for i in range(N):
    plots += [
        scipy.signal.resample(wav_raws[i],    PLOT_WAV_SAMPLE_NUM),
        scipy.signal.resample(wav_raw_dbs[i], PLOT_DB_SAMPLE_NUM),
        scipy.signal.resample(wav_outs[i],    PLOT_WAV_SAMPLE_NUM),
        scipy.signal.resample(wav_out_dbs[i], PLOT_DB_SAMPLE_NUM),
        ]

fig = plt.figure(figsize=(16,12))

for i in range(N):
    ax1 = fig.add_subplot(N, 4, 1 + 4*i)
    ax1.set_title(wav_paths[i])
    ax1.set_autoscale_on(False)
    ax1.axis([0, PLOT_WAV_SAMPLE_NUM, -PLOT_WAV_MAX, PLOT_WAV_MAX])
    ax1.plot(plots[0 + 4*i])

    ax2 = fig.add_subplot(N, 4, 2 + 4*i)
    ax2.set_title('input db')
    ax2.set_autoscale_on(False)
    ax2.axis([0, PLOT_DB_SAMPLE_NUM, PLOT_DB_MIN, PLOT_DB_MAX])
    ax2.plot(plots[1 + 4*i])

    ax3 = fig.add_subplot(N, 4, 3 + 4*i)
    ax3.set_title(out_paths[i])
    ax3.set_autoscale_on(False)
    ax3.axis([0, PLOT_WAV_SAMPLE_NUM, -PLOT_WAV_MAX, PLOT_WAV_MAX])
    ax3.plot(plots[2 + 4*i])

    ax4 = fig.add_subplot(N, 4, 4 + 4*i)
    ax4.set_title('output db')
    ax4.set_autoscale_on(False)
    ax4.axis([0, PLOT_DB_SAMPLE_NUM, PLOT_DB_MIN, PLOT_DB_MAX])
    ax4.plot(plots[3 + 4*i])

plt.savefig(PLOT_PATH)
