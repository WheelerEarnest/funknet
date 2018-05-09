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

def gradient_descent(learning_rate=0.01):
  """

  :param learning_rate: learning rate of the gradient descent
  :return: Returns a gradient for fetching
  """
  with tf.variable_scope('lstm', reuse=True):
    wf = tf.get_variable('wf', dtype=tf.complex64)
    bf = tf.get_variable('bf', dtype=tf.complex64)
    wu = tf.get_variable('wu', dtype=tf.complex64)
    bu = tf.get_variable('bu', dtype=tf.complex64)
    wo = tf.get_variable('wo', dtype=tf.complex64)
    bo = tf.get_variable('bo', dtype=tf.complex64)
    wc = tf.get_variable('wc', dtype=tf.complex64)
    bc = tf.get_variable('bc', dtype=tf.complex64)
    wy = tf.get_variable('wy', dtype=tf.complex64)
    by = tf.get_variable('by', dtype=tf.complex64)

    dwy = sum(tf.get_collection('dwy', 'lstm'))
    dby = sum(tf.get_collection('dby', 'lstm'))
    dwo = sum(tf.get_collection('dwo', 'lstm'))
    dbo = sum(tf.get_collection('dbo', 'lstm'))
    dwf = sum(tf.get_collection('dwf', 'lstm'))
    dbf = sum(tf.get_collection('dbf', 'lstm'))
    dwu = sum(tf.get_collection('dwu', 'lstm'))
    dbu = sum(tf.get_collection('dbu', 'lstm'))
    dwc = sum(tf.get_collection('dwc', 'lstm'))
    dbc = sum(tf.get_collection('dbc', 'lstm'))

    dwy = tf.clip_by_value(dwy, 10+10j, -10-10j)
    dby = tf.clip_by_value(dby, 10+10j, -10-10j)
    dwo = tf.clip_by_value(dwo, 10+10j, -10-10j)
    dbo = tf.clip_by_value(dbo, 10+10j, -10-10j)
    dwf = tf.clip_by_value(dwf, 10+10j, -10-10j)
    dbf = tf.clip_by_value(dbf, 10+10j, -10-10j)
    dwu = tf.clip_by_value(dwu, 10+10j, -10-10j)
    dbu = tf.clip_by_value(dbu, 10+10j, -10-10j)
    dwc = tf.clip_by_value(dwc, 10+10j, -10-10j)
    dbc = tf.clip_by_value(dbc, 10+10j, -10-10j)

    tf.assign_sub(wy, learning_rate * dwy)
    tf.assign_sub(by, learning_rate * dby)
    tf.assign_sub(wo, learning_rate * dwo)
    tf.assign_sub(bo, learning_rate * dbo)
    tf.assign_sub(wf, learning_rate * dwf)
    tf.assign_sub(bf, learning_rate * dbf)
    tf.assign_sub(wu, learning_rate * dwu)
    tf.assign_sub(bu, learning_rate * dbu)
    tf.assign_sub(wc, learning_rate * dwc)
    tf.assign_sub(bc, learning_rate * dbc)

    graph = tf.get_default_graph()

    graph.clear_collection('dwy')
    graph.clear_collection('dby')
    graph.clear_collection('dwo')
    graph.clear_collection('dbo')
    graph.clear_collection('dwf')
    graph.clear_collection('dbf')
    graph.clear_collection('dwu')
    graph.clear_collection('dbu')
    graph.clear_collection('dwc')
    graph.clear_collection('dbc')

    return dbc