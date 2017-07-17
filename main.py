import wave
import scipy
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from functools import reduce

VERBOSE = False
PLOT = True

IN_PATHS = ['./input/%d.wav' % i for i in range(0, 10)]
OUT_PATHS = [path.replace('./input/', './output/') for path in IN_PATHS]
PLOT_PATH = 'output/fig.png'

BIT = 32 - 1
SAMPLE_RATE = 41100
AGC_FRAME_MS = 10 #ms
AGC_SUB_FRAME_MS = 1 #ms
CHUNK_SIZE = int(41100 * AGC_FRAME_MS / 1000)
VAD_THES = 0.001
AGC_WINDOW = AGC_FRAME_MS * 1000

P_REF = 1 / 2 ** 14

AGC_DBS  = [-0. , -0.5,  -1., -1.5,  -2., -2.5,  -3., -3.5,  -4., -4.5,  -5., -5.5,  -6., -6.5,  -7., -7.5, -8.]
AGC_GAIN = [  -3,   -2,   -2,   -2,   -1,   -1,   -1,   -1,    0,    0,    0,    1,    2,    3,    4,    5,   6]

PLOT_WAV_SAMPLE_NUM = 10000
PLOT_WAV_MAX = 1.0
PLOT_DB_SAMPLE_NUM = 100
PLOT_DB_MIN = -20
PLOT_DB_MAX = 0
PLOT_GAIN_MAX = np.max(AGC_GAIN)
PLOT_GAIN_MIN = np.min(AGC_GAIN)
PLOT_PD_MAX = 2
PLOT_PD_MIN = 0
PLOT_VAD_MAX = 2
PLOT_VAD_MIN = 0

def wav_sample(li, k):
    ret = []
    step = int(len(li) / k)
    if step == 0:
        return li
    for i in range(0, len(li), step):
        ret.append(li[i])
    return np.array(ret)

def dBFS(wav):
    wav[wav == 0] = 0.0001
    return np.log2(np.abs(wav) / 1)

def trim_soundless(wav):
    return wav[wav > 0.001]

def mean_dBFS(wav):
    return np.mean(dBFS(np.trim_zeros(wav)))
    # return np.mean(dBFS(trim_soundless(wav)))

def peak_detector(wav):
    return np.max(np.multiply(wav, wav))

def voice_activity_detection(peaks):
    return np.mean(peaks[-10:]) > VAD_THES

def calc_gain_dBFS(wav, vad):
    db = mean_dBFS(wav)
    if not vad:
        return 0
    assert len(AGC_DBS) == len(AGC_GAIN), 'check'
    for i in range(len(AGC_DBS)):
        if db > AGC_DBS[i]:
            return AGC_GAIN[i]
    return AGC_GAIN[-1]


def geomtric_mean(li):
    ret = pow(reduce(lambda x,y:x*y, li), 1 / len(li))
    return ret if ret >= 0 else li[-1]

def arithmetic_mean(li):
    return np.mean(li)

def auto_gain_control(wav):
    chunks = np.array_split(wav, int(len(wav) / CHUNK_SIZE))
    wav_out = []
    gains = [0 for i in range(AGC_WINDOW)]
    wav_in_db = []
    wav_out_db = []
    peaks = []
    vads = []
    for chunk_in in chunks:
        peak = peak_detector(chunk_in)
        peaks.append(peak)

        vad = voice_activity_detection(peaks)
        vads.append(vad)

        
        if VERBOSE:        
            print('\n\tchunk: %.1f ms' %(len(chunk_in) / SAMPLE_RATE * 1000))
            print('\tAVG AMP: %.6f' % np.mean(np.abs(chunk_in)))
            print('\tPD: %.6f' % peak)
            print('\tVAD: %.6f' % vad)

        # AUTO GAIN CONTROL
        gain = calc_gain_dBFS(chunk_in, vad)
        # TODO 积分电路是这样实现的吗？
        gain = arithmetic_mean(gains[-AGC_WINDOW:] + [gain])
        if vad:
            gains.append(gain)
        chunk_out = chunk_in * (2 ** gain)
        wav_in_db.append(mean_dBFS(chunk_in))
        wav_out_db.append(mean_dBFS(chunk_out))
        wav_out.append(chunk_out)

        if VERBOSE: print('\tgain:%.4f db:%.2f -> %.2f' %(gain, mean_dBFS(chunk_in), mean_dBFS(chunk_out)))

    return np.concatenate(wav_out), np.array(gains[AGC_WINDOW:]), np.array(wav_in_db), np.array(wav_out_db), np.array(peaks), np.array(vads)

def AGC(wav):
    wav_outs = []
    wav_gains = [0 for i in range(AGC_WINDOW)]
    wav_in_dbs = []
    wav_out_dbs = []
    chunks = np.array_split(wav, 263)
    for chunk_in in chunks:
        pass
    return np.concatenate(wav_outs), wav_gains[AGC_WINDOW:], np.array(wav_in_dbs), np.array(wav_out_dbs)


N = len(IN_PATHS)
rates = [None for i in range(N)]
wav_in = [None for i in range(N)]
wav_outs = [None for i in range(N)]
wav_out_dbs  = [None for i in range(N)]
wav_in_dbs  = [None for i in range(N)]
wav_gains = [None for i in range(N)]
wav_peaks = [None for i in range(N)]
wav_vads = [None for i in range(N)]
for i in range(N):
    print('\nauto_gain_control:', IN_PATHS[i])

    # Input
    rates[i], wav_in[i] = scipy.io.wavfile.read(IN_PATHS[i])
    
    # Sample
    # wav_in[i] = wav_in[i][:SAMPLE_RATE * 2]

    # AGC
    wav_outs[i], wav_gains[i], wav_in_dbs[i], wav_out_dbs[i], wav_peaks[i], wav_vads[i]  = auto_gain_control(wav_in[i])

    # Output
    scipy.io.wavfile.write(OUT_PATHS[i], rates[i], wav_outs[i])

##### Graph ####
if not PLOT:
    exit()
print('\nplotting:', PLOT_PATH)
plots = []
for i in range(N):
    plots += [
        wav_sample(wav_in[i], PLOT_WAV_SAMPLE_NUM),
        wav_in_dbs[i],
        wav_peaks[i],
        wav_vads[i],
        wav_sample(wav_outs[i], PLOT_WAV_SAMPLE_NUM),
        wav_out_dbs[i],
        wav_gains[i],
        ]
FIG = 7
COL = 1
ROW = N * FIG
fig = plt.figure(figsize=(20 * COL, 3 * ROW))
for i in range(N):
    j = 0
    plot = plots[j + FIG * i]
    ax = fig.add_subplot(ROW, COL, 1 + j + FIG * i)
    ax.set_title(IN_PATHS[i])
    ax.set_autoscale_on(False)
    ax.axis([0, len(plot), -PLOT_WAV_MAX, PLOT_WAV_MAX])
    ax.plot(plot)

    j += 1
    plot = plots[j + FIG * i]
    ax = fig.add_subplot(ROW, COL, 1 + j + FIG * i)
    ax.set_title('input db')
    ax.set_autoscale_on(False)
    ax.axis([0, len(plot), PLOT_DB_MIN, PLOT_DB_MAX])
    ax.plot(plot)

    j += 1
    plot = plots[j + FIG * i]
    ax = fig.add_subplot(ROW, COL, 1 + j + FIG * i)
    ax.set_title('PD')
    ax.set_autoscale_on(False)
    ax.axis([0, len(plot), PLOT_PD_MIN, PLOT_PD_MAX])
    ax.plot(plot)


    j += 1
    plot = plots[j + FIG * i]
    ax = fig.add_subplot(ROW, COL, 1 + j + FIG * i)
    ax.set_title('VAD')
    ax.set_autoscale_on(False)
    ax.axis([0, len(plot), PLOT_VAD_MIN, PLOT_VAD_MAX])
    ax.plot(plot)

    j += 1
    plot = plots[j + FIG * i]
    ax = fig.add_subplot(ROW, COL, 1 + j + FIG * i)
    ax.set_title(OUT_PATHS[i])
    ax.set_autoscale_on(False)
    ax.axis([0, len(plot), -PLOT_WAV_MAX, PLOT_WAV_MAX])
    ax.plot(plot)

    j += 1
    plot = plots[j + FIG * i]
    ax = fig.add_subplot(ROW, COL, 1 + j + FIG * i)
    ax.set_title('output db')
    ax.set_autoscale_on(False)
    ax.axis([0, len(plot), PLOT_DB_MIN, PLOT_DB_MAX])
    ax.plot(plot)

    j += 1
    plot = plots[j + FIG * i]
    ax = fig.add_subplot(ROW, COL, 1 + j + FIG * i)
    ax.set_title('auto gain')
    ax.set_autoscale_on(False)
    ax.axis([0, len(plot), PLOT_GAIN_MIN, PLOT_GAIN_MAX])
    ax.plot(plot)

plt.savefig(PLOT_PATH, bbox_inches='tight', dpi=50)
