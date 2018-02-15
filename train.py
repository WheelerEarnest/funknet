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

