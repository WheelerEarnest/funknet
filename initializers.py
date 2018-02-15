#----------------------------------------------------------
# Author: Wheeler Earnest
#
# Project: FunkNet
#
# Description: Contains initializers suitable for complex weights
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
#
#   I heavily referenced
# https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/python/ops/init_ops.py
#------------------------------------------------------------------------------------------------------

import tensorflow as tf
from tensorflow.python.ops.random_ops import random_uniform
from tensorflow.python.ops.init_ops import Initializer

class complex_random_uniform(Initializer):


  def __init__(self, minval=-1.0, maxval=1.0, seed=None, dtype=tf.complex64):
    self.minval = minval
    self.maxval = maxval
    self.seed = seed
    self.dtype = dtype

  def __call__(self, shape, dtype=tf.complex64, partition_info=None):
    real = random_uniform(shape, self.minval, self.maxval, seed=self.seed)
    imag = random_uniform(shape, self.minval, self.maxval, seed=self.seed)
    return tf.complex(real, imag)

  def get_config(self):
    return {
      'minval': self.minval,
      'maxval': self.maxval,
      'seed': self.seed,
      'dtype': self.dtype.name
    }
