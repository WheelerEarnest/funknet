#----------------------------------------------------------
# Author: Wheeler Earnest
#
# Project: FunkNet
#
# Description: 
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

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.training import adam

class complex_adam(adam.AdamOptimizer):
  def _valid_dtypes(self):
    return set(
      [dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64, dtypes.complex64])