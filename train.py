#----------------------------------------------------------
# Author: Wheeler Earnest
#
# Project: FunkNet
#
# Description: Where the training of the net happens, still trying to work out kinks
#
# Sources: When writing this I consulted the paper "Deep Complex Networks"
#   by Trabelsi et al. for information regarding activations and weight
#   initializations
#
#   Inspired by https://github.com/jisungk/deepjazz
#
#   Or to be more specific, an assignment in the deep learning
#   specialization on Coursera: 
#   https://www.coursera.org/learn/nlp-sequence-models
#------------------------------------------------------------------------------------------------------

import numpy as np
import tensorflow as tf
from lstm import *
from tensorflow.contrib.signal import *
from tensorflow.contrib.ffmpeg import encode_audio, decode_audio

tf.reset_default_graph()

frame_length = 1024
frame_step = 512
# Read the file and perform stft
song = tf.read_file('inputs/153337.mp3.wav')
waveform = decode_audio(song, 'wav', samples_per_second=44100, channel_count=1)
decomp = stft(tf.transpose(waveform), frame_length, frame_step)

# Get rid of the channel dimension and flip it, so its a series of vectors
inputs = tf.transpose(tf.squeeze(decomp))
shape = 2583

# Initialize the weights and get the outputs
init_lstm(shape, shape)
outputs = complex_lstm_forward(inputs, tf.zeros([shape, 1], dtype=tf.complex64), tf.zeros([shape, 1], dtype=tf.complex64))
outputs = outputs.gather([outputs.size()])

new_decomp = tf.reshape(tf.transpose(outputs), [1, frame_length // 2 + 1, shape])
new_song = inverse_stft(new_decomp, frame_length, frame_step)
song = encode_audio(new_song, 'wav', samples_per_second=44100)
write = tf.write_file('crap.wav', song)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(write)