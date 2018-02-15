# ----------------------------------------------------------
# Author: Wheeler Earnest
#
# Project: FunkNet
#
# Description: Gathers up the songs, transforms them and puts them into a TensorArray
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
# ------------------------------------------------------------------------------------------------------

import tensorflow as tf
from os import listdir
from tensorflow.contrib.ffmpeg import encode_audio, decode_audio
from tensorflow.contrib.signal import stft, inverse_stft

def get_songs(folder, sample_rate):
  """
  Gather up all the singy-songs you want to train the net on
  :param sample_rate: An integer representing the samples per second
  :param folder: String of the path to the folder containing the data
  :return: returns a TensorArray with the decoded audio
  """
  files = listdir(folder)
  songs = tf.TensorArray(tf.float32, size=len(files))

  for file_name, i in zip(files, range(len(files))):
    file = tf.read_file(file_name)
    # We set the channel count to 1 because I am unsure how to make more work
    waveform = decode_audio(file, 'wav', sample_rate, channel_count=1)
    songs = songs.write(i, waveform)
  return songs

def decompose(waveform_list, frame_length, frame_step):
  """
  Decomposes the decoded audio into
  :param waveform_list: A TensorArray of all your decoded audio
  :param frame_length: Size of the window for the transform, in samples
  :param frame_step: Number of samples to step forward
  :return: returns a TensorArray of the decomposed audio
  """
