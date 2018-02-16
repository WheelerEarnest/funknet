# ----------------------------------------------------------
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
# ------------------------------------------------------------------------------------------------------

import numpy as np
import tensorflow as tf
from lstm import *
from tensorflow.contrib.signal import *
from tensorflow.contrib.ffmpeg import encode_audio, decode_audio


def cost(training_data, model_output):
  """
  In this method, we are looping through the TensorArray, which contains the training examples
  :param training_data: TensorArray containing the input data used for this batch
  :param model_output: TensorArray containing the output of the model
  :return: Returns the cost for these inputs
  """

  J = tf.zeros([1])
  def body(i, x, y_hat, J):
    J = J + loss(x.read(i), y_hat.read(i))
    return i + 1, x, y_hat, J

  def cond(i, x, y_hat, J):
    return i < x.size()

  __, __, __, J = tf.while_loop(cond, body, (0, training_data, model_output, J))

  return J

def loss(x, y_hat):
  """
  Computes the loss for one training example
  :param x:
  :param y_hat:
  :return:
  """
  # We dont care about the first input or the last output, as we are comparing the output
  #   to the input of the next timestep
  y_hat = y_hat[:-1,:]
  x = x[1:,:]
  diff = y_hat - x
  return tf.complex_abs(diff)



