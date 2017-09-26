#!/usr/bin/python

'''
Author : Karan Singla

Purpose : Give audio folder/file, extract audio features

Usage :

extractaudiofeaturefolder(<audio_filder>,<out_file>)

extractaudiofeaturefile

NOTE : as we are using pyadioanalysis, it only runs on python 2.7

'''
import glob,os
from subprocess import call
import _pickle as cPickle
from pyAudioAnalysis2 import audioBasicIO
from pyAudioAnalysis2 import audioFeatureExtraction

def extractaudiofeaturefolder(audio_folder, out_folder, frame_size=50, frame_step=25):
	'''
	takes folder as a input and extract audio features for it
	'''
	#convert into seconds from ms
	frame_size = float(frame_size)/1000
	frame_step = float(frame_step)/1000

	audio_files = glob.glob(audio_folder+"/*")
	for file in audio_files:
		file_id = os.path.basename(file).split(".")[0]
		features = extractaudiofeaturefile(file, frame_size=frame_size, frame_step=frame_step)
		cPickle.dump(features,open(out_folder+file_id+".afeats",'wb'))


def extractaudiofeaturefile(filename, frame_size=50, frame_step=25):

#	print frame_size,frame_step
	print("filename :",filename)
	call("sox --i "+filename+" > ./temp.inf",shell=True)
	lines = open('./temp.inf','r').readlines()[2]
	channels = int(lines.strip()[-1])
	print("channels :",channels)

	if channels == 1:
		call("cp "+filename+" temp.wav",shell=True)
	else:
		call("sox "+filename+" temp.wav remix 1,2",shell=True)
	[Fs, x] = audioBasicIO.readAudioFile("./temp.wav")
	F = audioFeatureExtraction.stFeatureExtraction(x, Fs, frame_size*Fs, frame_step*Fs);
	call("rm ./temp.wav",shell=True)
	return F
	'''
	takes file as a input and extracts audio features for it
	'''
