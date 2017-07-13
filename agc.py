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

IN_PATHS = ['./input/%d.wav' % i for i in range(1, 10)]
OUT_FILES = [path.replace('./input/', './output/') for path in IN_PATHS]
PLOT_PATH = 'output/fig.png'

PLOT_WAV_SAMPLE_NUM = 10000
PLOT_WAV_MAX = 1.0
PLOT_DB_SAMPLE_NUM = 100
PLOT_DB_MIN = -10
PLOT_DB_MAX = 0
PLOT_GAIN_MAX = 2
CHUNK_SIZE = 50
P_REF = 1 / 2 ** 14
BIT = 32 - 1

AGC_DBS  = [-0. , -0.5,  -1., -1.5,  -2., -2.5,  -3., -3.5,  -4., -4.5,  -5., -5.5,  -6., -6.5,  -7., -7.5, -8.]
AGC_GAIN = [  -3,   -2,   -2,   -2,   -1,   -1,   -1,   -1,    0,    0,    0,    1,    1,    1,    0,    0,   0]

def wav_sample(li, k):
    ret = []
    step = int(len(li) / k)
    if step == 0:
        return li
    for i in range(0, len(li), step):
        ret.append(li[i])
    return np.array(ret)

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

def trim_soundless(wav):
    return wav[wav > 0.001]

def mean_dBFS(wav):
    return np.mean(dBFS(np.trim_zeros(wav)))
    # return np.mean(dBFS(trim_soundless(wav)))

def auto_gain_control(wav):

    def peak_detector(chunk):
        return np.max(np.abs(chunk))

    def voice_activity_detection(peak):
        return peak > (2 ** 5) / (2 ** BIT)

    def calc_gain_dBFS(chunk):
        db = mean_dBFS(chunk)
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

    # Part
    AGC_WINDOW = 50
    chunks = np.array_split(wav, 263)
    ret = []
    wav_gains = [0 for i in range(AGC_WINDOW)]
    for chunk in chunks:
        if VERBOSE:        
            print('\n\tchunk:', len(chunk), chunk)
            print('\tabs avg:%.6f' % np.mean(np.abs(chunk)))
            # print('\tpeak:%.6f' % peak_detector(chunk))
            # print('\tact:',voice_activity_detection(peak))

        # AUTO GAIN CONTROLLER
        gain = calc_gain_dBFS(chunk)
        # TODO 积分电路是这样实现的吗？
        # gain = arithmetic_mean(wav_gains[-AGC_WINDOW:] + [gain])
        wav_gains.append(gain)
        ret.append(chunk * (2 ** gain))
        if VERBOSE: print('\tgain:%.4f db:%.2f -> %.2f' %(gain, mean_dBFS(chunk), mean_dBFS(chunk * (2 ** gain))))

    return np.concatenate(ret), wav_gains[AGC_WINDOW:]

N = len(IN_PATHS)
in_paths = IN_PATHS
out_paths = OUT_FILES
rates = [None for i in range(N)]
wav_raws = [None for i in range(N)]
wav_outs = [None for i in range(N)]
wav_out_dbs  = [None for i in range(N)]
wav_raw_dbs  = [None for i in range(N)]
wav_gains = [None for i in range(N)]
for i in range(N):
    print('\nauto_gain_control:', in_paths[i])
    rates[i], wav_raws[i] = scipy.io.wavfile.read(in_paths[i])
    wav_outs[i], wav_gains[i] = auto_gain_control(wav_raws[i])
    wav_out_dbs[i] = dBFS(wav_outs[i])
    wav_raw_dbs[i] = dBFS(wav_raws[i])

    scipy.io.wavfile.write(out_paths[i], rates[i], wav_outs[i])

##### Graph ####
print('\nplotting:', PLOT_PATH)
plots = []
for i in range(N):
    plots += [
        wav_sample(wav_raws[i], PLOT_WAV_SAMPLE_NUM),
        wav_raw_dbs[i],
        wav_sample(wav_outs[i], PLOT_WAV_SAMPLE_NUM),
        wav_out_dbs[i],
        wav_gains[i],
        ]
FIG = 5
COL = 1
ROW = N * FIG
fig = plt.figure(figsize=(8 * COL, 3 * ROW))
for i in range(N):
    ax1 = fig.add_subplot(ROW, COL, 1 + FIG * i)
    ax1.set_title(in_paths[i])
    ax1.set_autoscale_on(False)
    ax1.axis([0, PLOT_WAV_SAMPLE_NUM, -PLOT_WAV_MAX, PLOT_WAV_MAX])
    ax1.plot(plots[0 + FIG * i])

    ax2 = fig.add_subplot(ROW, COL, 2 + FIG * i)
    ax2.set_title('input db')
    ax2.set_autoscale_on(False)
    ax2.axis([0, len(plots[1 + FIG * i]), PLOT_DB_MIN, PLOT_DB_MAX])
    ax2.plot(plots[1 + FIG * i])

    ax3 = fig.add_subplot(ROW, COL, 3 + FIG * i)
    ax3.set_title(out_paths[i])
    ax3.set_autoscale_on(False)
    ax3.axis([0, PLOT_WAV_SAMPLE_NUM, -PLOT_WAV_MAX, PLOT_WAV_MAX])
    ax3.plot(plots[2 + FIG * i])

    ax4 = fig.add_subplot(ROW, COL, 4 + FIG * i)
    ax4.set_title('output db')
    ax4.set_autoscale_on(False)
    ax4.axis([0, len(plots[3 + FIG * i]), PLOT_DB_MIN, PLOT_DB_MAX])
    ax4.plot(plots[3 + FIG * i])

    ax5 = fig.add_subplot(ROW, COL, 5 + FIG * i)
    ax5.set_title('auto gain')
    ax5.set_autoscale_on(False)
    ax5.axis([0, len(plots[4 + FIG * i]), -PLOT_GAIN_MAX, PLOT_GAIN_MAX])
    ax5.plot(plots[4 + FIG * i])

plt.savefig(PLOT_PATH, bbox_inches='tight', dpi=70)
