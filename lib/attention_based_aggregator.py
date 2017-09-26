#!/usr/bin/python

''' Author : Karan Singla '''

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

#library imports
import numpy as np

#tensorflow imports
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.contrib import rnn
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

from lib.util import _attn_mul_fun, map_fn_mult, gather_axis


def BiRNN(lstm_fw_cell, lstm_bw_cell, x, sequence_length=None,idd='sent'):

	'''
	Input Variables
	lstm_bw_cell : 
	x : 
	sequence_length : 
	idd :
	'''

	# Get lstm cell output
	'''
	with tf.variable_scope(idd+'lstm1') as vs:
		outputs, states = tf.nn.dynamic_rnn(lstm_bw_cell, x, dtype=tf.float32, sequence_length=sequence_length)
		rnn_variables = [v for v in tf.all_variables()
						if v.name.startswith(vs.name)]
	'''
	with tf.variable_scope(idd+'lstm1') as vs:

		(fw_outputs,bw_outputs),(fw_state,bw_state) = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
														  cell_bw=lstm_bw_cell,
														  inputs=x,
														  dtype=tf.float32,
														  sequence_length=sequence_length,
														  parallel_iterations=64)
		outputs = tf.concat((fw_outputs, bw_outputs), 2)

		rnn_variables = [v for v in tf.all_variables()
						if v.name.startswith(vs.name)]


	return outputs,rnn_variables

def RNN(lstm_bw_cell, x, sequence_length=None,idd='sent'):

	'''
	Input Variables
	lstm_bw_cell : 
	x : 
	sequence_length : 
	idd :
	'''

	# Get lstm cell output
	with tf.variable_scope(idd+'lstm1') as vs:
		outputs, states = tf.nn.dynamic_rnn(lstm_bw_cell, x, dtype=tf.float32, sequence_length=sequence_length,time_major=True)
		rnn_variables = [v for v in tf.all_variables()
						if v.name.startswith(vs.name)]

	return outputs,rnn_variables

class DocAggregator(object):

	def __init__(self, embedding_size=128, sent_attention_size=None, doc_attention_size=None,doc_embedding_size=None,\
		sent_embedding_size=None,sent_aggregator=None, sent_projection_size=128,
		lstm_layer=1, sent_lstm_layer=1,keep_prob=0.7, num_class=1, idd='doc', multiatt=True):

		self.embedding_size = embedding_size
		self.doc_embedding_size = doc_embedding_size
		self.sent_embedding_size = sent_embedding_size
		self.sent_attention_size = sent_attention_size
		self.doc_attention_size = doc_attention_size
		self.sent_aggregator = sent_aggregator
		self.lstm_layer = lstm_layer
		self.sent_projection_size = sent_projection_size
		self.idd = idd
		self.multiatt = multiatt

		self.keep_prob = keep_prob
		self.sent_lstm_layer = sent_lstm_layer

		self.doc_attention_aggregator = Aggregator(embedding_size=self.sent_projection_size,
				attention_size=self.doc_attention_size, n_hidden=self.doc_embedding_size, 
				lstm_layer=lstm_layer, num_class=num_class, idd='doc')

		self._initiate_doc_attention_aggregator()

		print("initiating sent embedding projection layer")
		self.sent_proj_weights = tf.Variable(tf.random_uniform([self.sent_embedding_size, self.sent_projection_size], -1.0, 1.0),
			name=self.idd+'_sent_proj_weights')
		self.proj_bias = tf.Variable(tf.zeros([self.sent_projection_size]), name=self.idd+'_proj_bias')

		self.sent_proj_variables = [self.sent_proj_weights, self.proj_bias]




	def _initiate_doc_attention_aggregator(self):

		'''
		doc = tf.placeholder(tf.int32, [None,max_doc_size,max_sent_len], name='doc')
		doc_embed = tf.nn.embedding_lookup(self.embeddings, doc)

		seq_len = tf.placeholder(tf.int32, [None,300], name='seq-len')
		doc_len = tf.placeholder(tf.int32, [None], name='doc-len')

		'''

		#check if sent_aggregator is None or not
		#if None it will initiate a sentence encoder
		if self.sent_aggregator == None:

			print("No sentence aggregator found")
			print("Initiating a sentence aggregator")

			self.sent_aggregator = Aggregator(embedding_size=self.embedding_size, attention_size=self.sent_attention_size,
				n_hidden=self.sent_embedding_size, lstm_layer=self.sent_lstm_layer,idd='sent')

		else:
			print("Using previously initiated sentence aggregator")
	
	def _initiate_sentence_aggregator(self, embed, seq_len):

		with tf.name_scope('AttentionBasedAggregator'):
			self.sent_aggregator = Aggregator(sequence_length=seq_len, embedding_size=self.embedding_size,
				attention_size=self.sent_attention_size, embed = embed, n_hidden=self.sent_embedding_size,\
				lstm_layer=1, keep_prob=0.7,idd='sent')


	def _inititate_doc_aggregator(self, embed, doc_len, doc_attention_size, num_class=None,
		lstm_layer=1, keep_prob=0.7):
		'''
		this is the heler function for initiate_doc_attention_aggregator()
		'''

		with tf.name_scope('DocAttentionBasedAggregator'):

			self.doc_attention_aggregator = Aggregator(sequence_length=doc_len,embedding_size=self.sent_embedding_size,
				attention_size=self.doc_attention_size, embed=embed, n_hidden=self.doc_embedding_size, lstm_layer=lstm_layer,
				keep_prob=keep_prob,idd=self.idd)

#			#if using multiattention framework
#			if self.multiatt == True :
#				self.doc_attention_aggregator.init_multiattention_aggregator_lstm(num_class=num_class)
#			else:
#				self.doc_attention_aggregator.init_attention_aggregator()

#			self.doc_attention_aggregator.init_attention_aggregator()

	def _calculate_sentence_encodings(self, doc_embed,seq_len, doc_len, keep_prob):


		doc_context = map_fn_mult(fn=self.sent_aggregator.calculate_attention_with_lstm_doc, 
								  arrays=[doc_embed, doc_len,seq_len],
								  parallel_iterations=30,
								  swap_memory=True)
#		doc_context = tf.map_fn(self.sent_aggregator.calculate_attention_with_lstm_tuple, (doc_embed,seq_len),
#			dtype=(tf.int32, tf.int32))
#		doc_context = tf.nn.dropout(doc_context, keep_prob)

		return doc_context

	def pred_fn(self, current_output):
		return tf.add(tf.matmul(current_output, self.sent_proj_weights), self.proj_bias)

	def calculate_document_vector(self,doc_embed, seq_len, doc_len,keep_prob=0.7):

		print("doc_embed :",doc_embed.shape)		
		self.doc_context = self._calculate_sentence_encodings(doc_embed=doc_embed,seq_len=seq_len, doc_len=doc_len, keep_prob=keep_prob)
		self.doc_context = tf.identity(self.doc_context, name='document_sentence_embeddingss')
		'''
		with tf.variable_scope('batch_norm') as vs:
			self.doc_context = batch_norm(self.doc_context)

			self.batch_norm_variables = [v for v in tf.all_variables()
										if v.name.startswith(vs.name)]

		'''
		self.doc_context = tf.nn.dropout(self.doc_context, keep_prob)

		print("doc_context :",self.doc_context.shape)
		self.doc_projected = tf.map_fn(self.pred_fn, self.doc_context)
		print("doc_projected :",self.doc_projected.shape)
#		doc_context = tf.identity(self.doc_context, name='document_sentence_embedding')

		print("DOC_context sentence encodings :",self.doc_context.shape)

		if self.multiatt == True:
			doc_vector = self.doc_attention_aggregator.caluculate_multiattention_with_lstm(self.doc_projected, doc_len,
				keep_prob=keep_prob)
			print("DOC_VECTOR :",doc_vector.shape)
		else:
#			doc_vector = math_ops.reduce_mean(self.doc_context, [1])
			doc_vector = self.doc_attention_aggregator.calculate_attention_with_lstm(self.doc_projected, doc_len,
				keep_prob=keep_prob)
			print("aggregated DOC_VECTOR :",doc_vector.shape)
#		doc_vector = self.doc_attention_aggregator.calculate_attention_with_lstm(doc_context, doc_len)

#		context_vector = tf.nn.dropout(doc_vector, keep_prob)

		return doc_vector


class Aggregator(object):


	def __init__(self,embedding_size=None, attention_size=None, sequence_length=None, n_hidden=100,
		lstm_layer=1, num_class=1, keep_prob=0.7,idd='sent', multiatt=False):
		
		self.idd = idd #name for this instance
		self.embedding_size = embedding_size #dimension of word-vectors/sentence-vectors 
		self.attention_size = attention_size #dimension of attention vector
		self.keep_prob = keep_prob #droupout keep proability
		self.attention_variables = []
		self.aggregator_variables = []
		self.num_class = num_class
		self.flag = 0
		self.lstm_layer = lstm_layer

		if lstm_layer != 0:
			# if lstm_layer == 1, then one  has to also give embed to initiate RNN
			# Define lstm cells with tensorflow
			# Forward direction cell
			print("HAHAHAHAHHAHAHHA",self.idd)
			self.n_hidden = n_hidden #hidden layer num of features if using lstm_layer=1

			with tf.variable_scope(self.idd+'backward') as vs:
				self.lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)
#				self.lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden/2, state_is_tuple=True)
#				self.lstm_bw_cell = tf.nn.rnn_cell.MultiRNNCell([self.lstm_bw_cell]*lstm_layer)


			with tf.variable_scope(self.idd+'forward') as vs:
				self.lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)
#				self.lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden/2, state_is_tuple=True)
#				self.lstm_fw_cell = tf.nn.rnn_cell.MultiRNNCell([self.lstm_fw_cell]*lstm_layer)

#			with tf.variable_scope(self.idd+'lstm1') as vs:
#				outputs, states = tf.nn.dynamic_rnn(self.lstm_bw_cell, embed, dtype=tf.float32,sequence_length=sequence_length)
#				rnn_variables = [v for v in tf.all_variables()
#								if v.name.startswith(vs.name)]

			print("Initiated Aggregator with LSTM layer")
#			self.aggregator_variables =  self.aggregator_variables + lstm_variables

		else:

			self.n_hidden = embedding_size #hidden layer num of features if using lstm_layer
			print("Initiated Aggregator without LSTM layer")

		self.attention_task = tf.Variable(tf.zeros([self.num_class, self.attention_size]),
			name=self.idd+'attention_vector')
		self.trans_weights = tf.Variable(tf.random_uniform([self.n_hidden, self.attention_size], -1.0, 1.0),
			name=self.idd+'transformation_weights')
		self.trans_bias = tf.Variable(tf.zeros([self.attention_size]), name=self.idd+'_trans_bias')

		self.attention_variables.append(self.attention_task)
		self.attention_variables.append(self.trans_weights)
		self.attention_variables.append(self.trans_bias)

		self.aggregator_variables = self.aggregator_variables + self.attention_variables

			# Backward direction cell
#			with tf.variable_scope('backward'):
#				self.lstm_fw_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)

	def zeros_average(self,keys,sequence_length):

		seq_range = tf.range(0,sequence_length,1)
		keys = tf.gather(tf.unstack(keys, axis=0), seq_range)

#		print(keys.shape)
		keys = math_ops.reduce_mean(keys, [0])
#		keys = tf.reshape(keys,[1,tf.shape(keys)[0]])
#		print(keys.shape)
#		pad_value = 60 - tf.shape(keys)[0]
#		t = [[0,pad_value,],[0,0]]
#		key = tf.pad(keys,t,"CONSTANT")

		return keys

	def average(self, embed,sequence_length):

		print(embed.shape)
		print(sequence_length.shape)
		context_vector = map_fn_mult(fn=self.zeros_average, 
								  arrays=[embed, sequence_length],
								  parallel_iterations=50,
								  swap_memory=True)
		print("Context woho",context_vector)

#		context_vector = math_ops.reduce_mean(embed, [1])
#		context_vector = tf.nn.dropout(context_vector, self.keep_prob)
		return context_vector 

	def average_with_lstm(self, embed,sequence_length):

		'''
		for using this method make sure
		that lstm_layer == 1 while initiating the 
		aggregator
		'''

		# get BiRNN outputs
		outputs,rnn_variables = RNN(self.lstm_bw_cell, embed, sequence_length,idd=self.idd)
#		outputs = tf.nn.dropout(outputs, self.keep_prob)
		if self.flag == 0:

			self.aggregator_variables = self.aggregator_variables + rnn_variables
			self.flag = self.flag + 1

		context_vector = math_ops.reduce_mean(outputs, [1])
#		context_vector = tf.nn.dropout(context_vector, self.keep_prob)
		return context_vector

	def calculate_multiattention(self,embed):

		embeddings_flat = tf.reshape(embed, [-1, self.n_hidden])

		print("multiattention : : ",self.attention_task)
		self.attention_task_unstack = tf.unstack(self.attention_task)
		print("multiattention len : : ",len(self.attention_task_unstack))
		# tanh transformation of embeddings
		keys_flat = tf.tanh(tf.add(tf.matmul(embeddings_flat,
			self.trans_weights), self.trans_bias))

		# reshape the keys according to our embed vector
		keys = tf.reshape(keys_flat, tf.concat(axis=0,values=[tf.shape(embed)[:-1], [self.attention_size]]))



		context_vectors = []

		for i in range(0,len(self.attention_task_unstack)):

			

			# Now calculate the attention-weight vector.

			# tanh transformation of embeddings
			keys_flat = tf.tanh(tf.add(tf.matmul(embeddings_flat,
				self.trans_weights), self.trans_bias))

			# reshape the keys according to our embed vector
			keys = tf.reshape(keys_flat, tf.concat(axis=0,values=[tf.shape(embed)[:-1], [self.attention_size]]))

			# calculate score for each word embedding and take softmax on it
			scores = math_ops.reduce_sum(keys * self.attention_task_unstack[i], [2])
			alignments = nn_ops.softmax(scores)

			# expand aligments dimension so that we can multiply it with embed tensor
			alignments = array_ops.expand_dims(alignments,2)

			alignments = tf.identity(alignments, name=self.idd+'_attention_weights')

			# generate context vector by making 
			context_vector = math_ops.reduce_sum(alignments * embed, [1])
#			context_vector = tf.nn.dropout(context_vector, self.keep_prob)
			context_vectors.append(context_vector)

		context_vectors = tf.stack(context_vectors)
		return context_vectors

	def caluculate_multiattention_with_lstm(self,embed, sequence_length, keep_prob):

		context_vectors = []

		# get BiRNN outputs
		outputs,rnn_variables = BiRNN(lstm_fw_cell=self.lstm_fw_cell,
									  lstm_bw_cell=self.lstm_bw_cell,
									  x=embed,
									  sequence_length=sequence_length,
									  idd=self.idd)

		if self.flag == 0:

			self.aggregator_variables = self.aggregator_variables + rnn_variables
			self.flag = self.flag + 1		
#		outputs = tf.nn.dropout(outputs, keep_prob)

		context_vectors = self.calculate_multiattention(outputs)
		return context_vectors

	def zeros_before_attention(self,keys,sequence_length):

		seq_range = tf.range(0,sequence_length,1)
		keys = tf.gather(tf.unstack(keys, axis=0), seq_range)

		print("zeros_keys",keys.shape)
		pad_value = 50 - tf.shape(keys)[0]
		t = [[0,pad_value,],[0,0]]
		key = tf.pad(keys,t,"CONSTANT")

		return key

	def zeros_softmax(self,scores,sequence_length):

		seq_range = tf.range(0,sequence_length,1)
		scores = tf.gather(tf.unstack(scores, axis=0), seq_range)
		scores = nn_ops.softmax(scores)
		pad_value = 50 - tf.shape(scores)[0]
		t = [[0,pad_value]]
		scores = tf.pad(scores,t,"CONSTANT")

		return scores

	def calculate_attention(self, embed,sequence_length):

		embeddings_flat = tf.reshape(embed, [-1, self.n_hidden])

		print("embedding flat shape",embeddings_flat.shape)
		# Now calculate the attention-weight vector.

		# tanh transformation of embeddings
		keys_flat = tf.tanh(tf.add(tf.matmul(embeddings_flat,
			self.trans_weights), self.trans_bias))

		# reshape the keys according to our embed vector
		keys = tf.reshape(keys_flat, tf.concat(axis=0,values=[tf.shape(embed)[:-1], [self.attention_size]]))

		print("keys shape",keys.shape)

		self.keys = tf.identity(keys,name=self.idd+"_keys")

		keys = tf.reshape(keys,[tf.shape(keys)[0],50,self.attention_size])

	
		print("keys shape",keys.shape)

		keys = map_fn_mult(fn=self.zeros_before_attention, 
								  arrays=[keys, sequence_length],
								  parallel_iterations=50,
								  swap_memory=True)

#		self.keys_with_zeros = tf.identity(keys,name=self.idd+"_keys_with_zeros")
		# calculate score for each word embedding and take softmax on it
		self.scores = math_ops.reduce_sum(keys * self.attention_task, [2])
		self.scores = tf.reshape(self.scores,[tf.shape(self.scores)[0],50])
		print("scores shape",self.scores.shape)
		self.alignments_softmax = map_fn_mult(fn=self.zeros_softmax, 
								  arrays=[self.scores, sequence_length],
								  parallel_iterations=50,
								  swap_memory=True)

		self.alignments_softmax = nn_ops.softmax(self.scores)

		# expand aligments dimension so that we can multiply it with embed tensor
		self.alignments = array_ops.expand_dims(self.alignments_softmax,2)

		self.alignments = tf.identity(self.alignments, name=self.idd+'_attention_weights')

		# generate context vector by making 
		context_vector = math_ops.reduce_sum(self.alignments * embed, [1])
#		context_vector = tf.nn.dropout(context_vector, self.keep_prob)
		return context_vector,self.alignments_softmax

	def calculate_attention_with_lstm(self, embed, sequence_length=None, keep_prob=1.0):

		'''
		this method only works if you use
		init_attention_aggregator_lstm
		sequence_length : 1D matrix having original length of sentences of inputs in x
		'''
		print(embed.shape)

		# get BiRNN outputs
		if self.lstm_layer != 0:
			print("embed shape", embed.shape)
			outputs,rnn_variables = BiRNN(self.lstm_bw_cell, embed, sequence_length,idd=self.idd)

			print("rnn outputs shape2 :",outputs.shape)

		else:
			outputs = embed
			rnn_variables = []
			print("No sentence level lstm layer")

		print("rnn outputs shape :",outputs.shape)
		if self.flag == 0:

			self.aggregator_variables = self.aggregator_variables + rnn_variables
			self.flag = self.flag + 1
#		outputs = tf.nn.dropout(outputs, keep_prob)
		
		outputs = tf.transpose(outputs,perm=[1,0,2])
		slices = []
		for index, l in enumerate(tf.unstack(sequence_length)):
			slic = tf.slice(outputs, begin=[index, l - 1, 0], size=[1, 1, 128])
			slices.append(slic)
		print("slices",slices)
		last = tf.concat(slices,0)
#		context_vector = tf.transpose(outputs,perm=[1,0,2])
		context_vector = tf.squeeze(last)
		print("returned context_vector")
		'''
		context_vector,alignments = self.calculate_attention(outputs,sequence_length=sequence_length)
		return context_vector,alignments
		'''
		return context_vector,outputs
	def calculate_attention_with_lstm_doc(self, embed, doc_len, sequence_length,keep_prob=1.0):

		'''
		this method only works if you use
		init_attention_aggregator_lstm
		sequence_length : 1D matrix having original length of sentences of inputs in x
		'''
		if self.attention_task is None:
			print("Initiating attention mechanism with lstm")
			init_attention_aggregator_lstm()

		# get BiRNN outputs
		if self.lstm_layer != 0:

			outputs,rnn_variables = BiRNN(lstm_fw_cell=self.lstm_fw_cell,
									  lstm_bw_cell=self.lstm_bw_cell,
									  x=embed,
									  sequence_length=sequence_length,
									  idd=self.idd)
			print("rnn outputs shape2 :",outputs.shape)

		else:
			outputs = self.average(embed,sequence_length)
			rnn_variables = []
			print("No sentence level lstm layer")
		
		if self.flag == 0:

			self.aggregator_variables = self.aggregator_variables + rnn_variables
			self.flag = self.flag + 1


#		doc_range = tf.range(0,doc_len,1)

#		outputs = tf.gather(tf.unstack(outputs, axis=0), doc_range)
#		print("few sentences shape :",outputs.shape)
#		outputs = tf.nn.dropout(outputs, keep_prob)
		'''
		embeddings_flat = tf.reshape(outputs, [-1, self.n_hidden])

		# tanh transformation of embeddings
		keys_flat = tf.tanh(tf.add(tf.matmul(embeddings_flat,
			self.trans_weights), self.trans_bias))

		# reshape the keys according to our embed vector
		keys = tf.reshape(keys_flat, tf.concat(axis=0,values=[tf.shape(outputs)[:-1], [self.attention_size]]))
		
		# calculate score for each word embedding and take softmax on it
		scores = _attn_mul_fun(keys, self.attention_task)
		alignments = nn_ops.softmax(scores)

		# expand aligments dimension so that we can multiply it with embed tensor
		alignments = array_ops.expand_dims(alignments,2)

		alignments = tf.identity(alignments, name=self.idd+'_attention_weights')

		# generate context vector by making 
		context_vector = math_ops.reduce_sum(alignments * outputs, [1])
		
		pad_value = 100 - tf.shape(context_vector)[0]
		t = [[0,pad_value,],[0,0]]
		context_vector = tf.pad(context_vector,t,"CONSTANT")

		print("context_vector_sent :", context_vector.shape)
#		context_vector = tf.nn.dropout(context_vector, keep_prob)
		return context_vector
		'''
		return outputs
