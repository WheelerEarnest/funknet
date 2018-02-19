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
  :param folder: String of the path to the folder containing the data, put a / at the end
  :return: returns a TensorArray with the decoded audio
  """
  files = listdir(folder)
  songs = tf.TensorArray(tf.float32, size=len(files))

  for file_name, i in zip(files, range(len(files))):
    file = tf.read_file(folder + file_name)
    # I set the channel count to 1 because I am unsure how to make more work
    waveform = decode_audio(file, 'wav', sample_rate, channel_count=1)
    songs = songs.write(i, waveform)
  return songs


def transform(waveform_list, frame_length, frame_step, fft_length=None):
  """
  Computes the short-time Fourier transform on each waveform
  Note: It's necessary to transpose the waveform before transforming, even though it's not mentioned in the tf docs.
  Heavily referenced: https://www.tensorflow.org/api_docs/python/tf/contrib/signal/stft
  :param waveform_list: A TensorArray of all your decoded audio
  :param frame_length: Size of the window for the transform, in samples (must be an integer scalar)
  :param frame_step: Number of samples to step forward (must be an integer scalar)
  :param fft_length: Size of the transform to apply, defaults to the smallest power of 2 enclosing frame_length
  :return: returns a TensorArray of the decomposed audio
  """
  # TensorArray that contains the transformed waveforms
  decomp = tf.TensorArray(tf.complex64, size=waveform_list.size())
  i = tf.constant(0)

  # Loop through the array and decompose each waveform
  def body(i, wave_list, decomp):
    transform = stft(tf.transpose(wave_list.read(i)), frame_length,
                     frame_step, fft_length)
    #Squeeze the transform to get rid of the channel dimension,
    # and transpose it, so that each frame is a vector
    transform = tf.transpose(tf.squeeze(transform))
    decomp = decomp.write(i, transform)
    return i+1, wave_list, decomp

  def cond(i, wave_list, decomp):
    return i < wave_list.size()

  __, __, decomp = tf.while_loop(cond, body, (i, waveform_list, decomp))
  return decomp

def reform(transform_list, frame_length, frame_step, fft_length=None):
  """
  Translates the output of the net into encodeable audio
  :param transform_list: TensorArray containing the transformed sound
  :param frame_length: Should be the same value that you used in the original transform
  :param frame_step: Should be the same value from the transform
  :param fft_length: Should be same value from transform
  :return: Returns a TensorArray of raw audio
  """
  waveform_list = tf.TensorArray(tf.float32, size=transform_list.size())
  i = tf.constant(0)

  def body(i, wave_list):
    # We need to reshape the outputs of the net.
    # It outputs tensor of size[frames, fft_bins, 1], whereas we want [1, frames, fft_bins]
    output = tf.squeeze(transform_list.read(i))[tf.newaxis, :, :]
    waveform = inverse_stft(output, frame_length, frame_step, fft_length)
    # Remember how we needed to transpose the raw audio before transforming it?
    # We need to do the same again, so that the audio can be encoded
    wave_list.write(i, tf.transpose(waveform))
    return i+1, wave_list

  def cond(i):
    return i < transform_list.size()

  __, waveform_list = tf.while_loop(cond, body, (i, waveform_list))
  return waveform_list

def save_songs(folder, songs, sample_rate):
  """
  Method to save the songs that have been generated
  :param folder: String representing the folder that the songs should be saved to.
       Please include a / at the end so that it works.
  :param songs: TensorArray that contains the endcodeable audio
  :param sample_rate: The sample rate for the new audio
  :return: Returns an integer and the write operation
  """
  tf.constant(0)

  def body(i, write):
    song_tensor = songs.read(i)
    encoded_song = encode_audio(song_tensor, 'wav', samples_per_second=sample_rate)
    write = tf.write_file(folder + str(i) + '.wav', encoded_song)
    return i+1, write

  def cond(i, write):
    return i < songs.size()

  __, write = tf.while_loop(cond, body, (i, None))
  return write


