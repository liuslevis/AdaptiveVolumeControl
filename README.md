# AdaptiveVolumeControl
Automatically control volume of songs in playlist to make a better experience.

## Introduction

## Data Prep

```
for ((i=1;i<10;i++)); do
    ffmpeg -i input/${i}.mp3 -acodec pcm_f32le -ac 1 -ar 44100 input/${i}.wav;
done
```

## Reference

Perez, Automatic Gain Control http://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=4&ved=0CEEQFjAD&url=http://www.springer.com/cda/content/document/cda_downloaddocument/9781461401667-c2.pdf?SGWID=0-0-45-1180039-p174121673&ei=LtcQU7McgrXgBMTdgdAI&usg=AFQjCNHUReiMDeUPA-hpYoerCXiPZLvXZg&sig2=j5UA1xMWoB-qjdelhsDpLQ&bvm=bv.62286460,d.bGE&cad=rja

TI: Software Implementation of Automatic Gain Controller for
Speech Signal http://www.ti.com/lit/wp/spraal1/spraal1.pdf

Automatic gain control https://en.wikipedia.org/wiki/Automatic_gain_control

Adaptive volume normalization on a stream of PCM data https://dsp.stackexchange.com/questions/24335/adaptive-volume-normalization-on-a-stream-of-pcm-data

Adaptive Gain Control with the Least Mean Squares Algorithm https://www.allaboutcircuits.com/technical-articles/adaptive-gain-control-with-the-least-mean-squares-algorithm/

Adaptive Volume Rendering using Fuzzy Logic
Control http://web.cse.ohio-state.edu/~shen.94/papers//Li2001.pdf

Automatic Loudness Control https://www.acoustics.asn.au/conference_proceedings/ICA2010/cdrom-ICA2010/papers/p960.pdf

StreamingKit https://github.com/login?return_to=%2Ftumtumtum%2FStreamingKit

iOS AVAudioEngine http://www.jianshu.com/p/506c62183763

Audio API Overview https://www.objc.io/issues/24-audio/audio-api-overview/

Understanding Automatic Gain Control https://www.allaboutcircuits.com/technical-articles/understanding-automatic-gain-control/

tf_agc - Time-frequency Automatic Gain Control https://labrosa.ee.columbia.edu/matlab/tf_agc/
