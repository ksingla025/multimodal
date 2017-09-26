#!/usr/bin/python


#standard python imports
import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')

#standard python imports
from collections import Counter
import math
import os
import random
import zipfile
import glob
import ntpath
import re
import random
from itertools import compress
import _pickle as cPickle
import pdb
from pathlib import Path
import pdb
import json

#library imports
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.contrib import rnn
from sklearn.base import BaseEstimator, TransformerMixin


# external library imports
#from utils.twokenize import *
from lib.path import *
from lib.util import *
from lib.attention_based_aggregator import *
from lib.skipgram import SkipGramGraph
from lib.batch_data_generators import generate_train_batch_data_task_leidos,\
generate_test_batch_data_task_leidos, generate_batch_data_task_ldcsf



class LSTMClassifier(object):
	'''
	this classes uses a pretrained LSTM encoder for getting word level feature and then trains
	at the top a classifier for prediction
	'''
	def __init(LSTMencoder=None, wordembedding_dim=64, sentembedding_dim=64,word_attention=False):

		self.LSTMencoder = LSTMencoder
		self.wordembedding_dim = wordembedding_dim
		self.sentembedding_dim = sentembedding_dim
		self.word_attention = word_attention


		sess=tf.Session()    
		#First let's load meta graph and restore weights
		saver = tf.train.import_meta_graph(LSTMencoder+'.meta')
		saver.restore(sess,tf.train.latest_checkpoint('./models/'))
 
 
		# Now, let's access and create placeholders variables and
		# create feed-dict to feed new data
 
		graph = tf.get_default_graph()

		self.input = graph.get_tensor_by_name("input_placeholder:0")
		enc_state = graph.get_tensor_by_name("enc_state:0")
		w2 = graph.get_tensor_by_name("w2:0")
		feed_dict ={w1:13.0,w2:17.0}
 
		#Now, access the op that you want to run. 
		op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
 
		#Add more to the current graph
		add_on_op = tf.multiply(op_to_restore,2)
 
		print sess.run(add_on_op,feed_dict)