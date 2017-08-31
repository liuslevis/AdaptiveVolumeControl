#!/usr/bin/python

import subprocess
import re
import sys

def DoFfmpeg(audiofile):
	cmd = 'ffmpeg -i %s -af "volumedetect" -f null /dev/null' % audiofile
	print('cmd:', cmd)
	p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=True)
	stdout, stderr = p.communicate()
	output = ""
	if p.returncode == 0:
		return (str(stdout) + str(stderr))

def DoAdjust(ffmpeg_output):
	mean_volume = 0
	max_volume = 0

	mean_volume_matches = re.findall(r"mean_volume: ([\-\d\.]+) dB", output)
	if mean_volume_matches:
		mean_volume = float(mean_volume_matches[0])
	else:
		raise ValueError("could not get mean volume")

	max_volume_matches = re.findall(r"max_volume: ([\-\d\.]+) dB", output)
	if max_volume_matches:
		max_volume = float(max_volume_matches[0])
	else:
		raise ValueError("could not get max volume")

	return min(-26.0 - mean_volume, 0 - max_volume), mean_volume, max_volume

if __name__ == '__main__':
	if len(sys.argv) == 1:
		print("usage: %s audiofile" % sys.argv[0])
		sys.exit()

	output = DoFfmpeg(sys.argv[1])
	adjust, mean_volume, max_volume = DoAdjust(output)
	
	print('mean dB:', mean_volume, 'max dB:', max_volume, 'adjust:', adjust)
