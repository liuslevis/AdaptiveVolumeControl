import wave
import scipy
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from functools import reduce

VERBOSE = True
# WAV_FILES = ['./input/%d.wav' % i for i in range(1,3)] + ['./input/3.mid.wav']#+ ['./input/3.mid.wav', './input/3.loud.wav']
WAV_FILES = ['./input/1.wav']
OUT_FILES = [path.replace('./input/', './output/') for path in WAV_FILES]
PLOT_PATH = 'output/fig.png'

PLOT_WAV_SAMPLE_NUM = 10000
PLOT_WAV_MAX = 0.1
PLOT_DB_SAMPLE_NUM = 100
PLOT_DB_MIN = -15
PLOT_DB_MAX = 0
PLOT_GAIN_MAX = 4
CHUNK_SIZE = 50
P_REF = 1 / 2 ** 14
BIT = 32 - 1


def dBFS(wav):
    def dBFS_(Prms):
        return np.log2(np.abs(Prms) / 1)
    ret = []
    chunks = np.array_split(wav, CHUNK_SIZE)
    chunks = list(map(lambda x:x.astype(np.float32), chunks))
    for chunk in chunks:
        prms = np.mean(np.sqrt(np.abs(chunk ** 2)))
        db = dBFS_(prms)
        ret.append(db)
    return np.array(ret)

def mean_dBFS(wav):
    return np.mean(dBFS(np.trim_zeros(wav)))


def auto_gain_control(wav):
    wav_gains = []

    def peak_detector(chunk):
        return np.max(np.abs(chunk))

    def voice_activity_detection(peak):
        return peak > (2 ** 5) / (2 ** BIT)

    def calc_gain_dBFS(chunk):
        db = mean_dBFS(chunk)
        gain = 0
        if db > 0:
            gain = 0
        elif db > -70:
            gain = - 1.5 * db - 9
        else:
            gain = 0
        return gain

    def geomtric_mean(li):
        ret = pow(reduce(lambda x,y:x*y, li), 1 / len(li))
        return ret if ret >= 0 else li[-1]

    def arithmetic_mean(li):
        return np.mean(li)

    # Part
    
    chunks = np.array_split(wav, 120)
    ret = []
    for chunk in chunks:
        # PD
        peak = peak_detector(chunk)
        # VAD
        act  = voice_activity_detection(peak)
        if VERBOSE:        
            print()
            print('\tchunk:', chunk)
            print('\tabs avg:%.6f' % np.mean(np.abs(chunk)))
            print('\tpeak:%.6f' % peak)
            print('\tact:',act)
        # GAIN CONTROLLER
        if act:
            gain = calc_gain_dBFS(chunk)
            # TODO 积分电路是这样实现的吗？
            gain = arithmetic_mean(wav_gains[-10:] + [gain])
            wav_gains.append(gain)
            ret.append(chunk * (2**gain))
            if VERBOSE: print('\tgain:%.4f db:%.2f -> %.2f' %(gain, mean_dBFS(chunk), mean_dBFS(chunk * (2**gain) )))
        else:
            ret.append(chunk)
            if VERBOSE: print('\tgain:0')

    return np.concatenate(ret), wav_gains

N = len(WAV_FILES)
wav_paths = WAV_FILES
out_paths = OUT_FILES
rates = [None for i in range(N)]
wav_raws = [None for i in range(N)]
wav_outs = [None for i in range(N)]
wav_out_dbs  = [None for i in range(N)]
wav_raw_dbs  = [None for i in range(N)]
wav_gains = [None for i in range(N)]
for i in range(N):
    rates[i], wav_raws[i] = scipy.io.wavfile.read(wav_paths[i])
    print('\nauto_gain_control:', wav_paths[i])
    wav_outs[i], wav_gains[i] = auto_gain_control(wav_raws[i])
    wav_out_dbs[i] = dBFS(wav_outs[i])
    wav_raw_dbs[i] = dBFS(wav_raws[i])

    scipy.io.wavfile.write(out_paths[i], rates[i], wav_outs[i])

##### Graph ####
print('\nplotting:', PLOT_PATH)
plots = []

for i in range(N):
    plots += [
        scipy.signal.resample(wav_raws[i],    PLOT_WAV_SAMPLE_NUM),
        scipy.signal.resample(wav_raw_dbs[i], PLOT_DB_SAMPLE_NUM),
        scipy.signal.resample(wav_outs[i],    PLOT_WAV_SAMPLE_NUM),
        scipy.signal.resample(wav_out_dbs[i], PLOT_DB_SAMPLE_NUM),
        wav_gains[i],
        ]

fig = plt.figure(figsize=(16, 4*N))

M = 5
for i in range(N):
    ax1 = fig.add_subplot(N, M, 1 + M*i)
    ax1.set_title(wav_paths[i])
    ax1.set_autoscale_on(False)
    ax1.axis([0, PLOT_WAV_SAMPLE_NUM, -PLOT_WAV_MAX, PLOT_WAV_MAX])
    ax1.plot(plots[0 + M*i])

    ax2 = fig.add_subplot(N, M, 2 + M*i)
    ax2.set_title('input db')
    ax2.set_autoscale_on(False)
    ax2.axis([0, PLOT_DB_SAMPLE_NUM, PLOT_DB_MIN, PLOT_DB_MAX])
    ax2.plot(plots[1 + M*i])

    ax3 = fig.add_subplot(N, M, 3 + M*i)
    ax3.set_title(out_paths[i])
    ax3.set_autoscale_on(False)
    ax3.axis([0, PLOT_WAV_SAMPLE_NUM, -PLOT_WAV_MAX, PLOT_WAV_MAX])
    ax3.plot(plots[2 + M*i])

    ax4 = fig.add_subplot(N, M, 4 + M*i)
    ax4.set_title('output db')
    ax4.set_autoscale_on(False)
    ax4.axis([0, PLOT_DB_SAMPLE_NUM, PLOT_DB_MIN, PLOT_DB_MAX])
    ax4.plot(plots[3 + M*i])

    ax5 = fig.add_subplot(N, M, 5 + M*i)
    ax5.set_title('auto gain')
    ax5.set_autoscale_on(False)
    ax5.axis([0, len(plots[4 + M*i]), -PLOT_GAIN_MAX, PLOT_GAIN_MAX])
    ax5.plot(plots[4 + M*i])

plt.savefig(PLOT_PATH)
