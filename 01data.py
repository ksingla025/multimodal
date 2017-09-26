#!/usr/bin/python

'''
Author : Karan Singla
Email : ksingla025@gmail.com
'''


from lib.audio_feature_extract import extractaudiofeaturefolder
from lib.data_utils import alignfolder2frame, cleanlabelfile, addaudiofeature2align, create_dictionary,labelinfofile2encoderdata
from subprocess import call
#paths
DATA = "./data/"
ALIGNMENT = DATA+"alignments/"
LABEL_FILE = DATA+"label/all-misc-global-data.txt"
PROCESSED = DATA+"processed/"
call("mkdir -p "+PROCESSED, shell=True)


AUDIO_FEATURES = PROCESSED+"/audio_feats/"
call("mkdir -p "+AUDIO_FEATURES, shell=True)

ALIGNMENT_CLEANED = PROCESSED+"/alignment_cleaned/"
call("mkdir -p "+ALIGNMENT_CLEANED, shell=True)


LABEL_CLEANED = PROCESSED+"./label_cleaned/"
call("mkdir -p "+LABEL_CLEANED, shell=True)

FRAME_SIZE = 50
FRAME_STEP = 25
#extract audio features

#extractaudiofeaturefolder(audio_folder='./data/audio/train', out_folder=AUDIO_FEATURES, frame_size=FRAME_SIZE, frame_step=FRAME_STEP)

#alignfolder2frame(align_folder=ALIGNMENT,out_folder=ALIGNMENT_CLEANED,frame_step=25)

#create_dictionary(filename=LABEL_FILE, out_file=PROCESSED+"dictionary.p", min_count=1)

#cleanlabelfile(clean_alignments=ALIGNMENT_CLEANED,label_file=LABEL_FILE, out_folder=LABEL_CLEANED,
#	audio_feature_folder=AUDIO_FEATURES, word2index=PROCESSED+'dictionary.p')

#addaudiofeature2align(clean_align=ALIGNMENT_CLEANED,audio_feature_file=PROCESSED+'audio_features.p',out_folder='asdas',sentlen=50,wordlen=50,features=34)

data,data_len = labelinfofile2encoderdata(data_folder=LABEL_CLEANED,out_file='out.temp')
print(len(data))
print("\n\n\n")
print(len(data_len))
