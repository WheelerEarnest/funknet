#----------------------------------------------------------
# Author: Wheeler Earnest
#
# Project: FunkNet
#
# Description: Contains the utility functions for creating an LSTM
#   that can handle complex inputs and complex weights
#
# Sources: I consulted the paper "Deep Complex Networks"
#   by Trabelsi et al. for information regarding activations and weight
#   initializations.
#
#   Inspired by https://github.com/jisungk/deepjazz
#
#   Or to be more specific, an assignment in the deep learning
#   specialization on Coursera:
#   https://www.coursera.org/learn/nlp-sequence-models
#-----------------------------------------------------------

import tensorflow as tf


def complex_lstm_cell(x, a_prev, c_prev):
  """

  :param x: Input vector at the current time-step. (n_x, m)
  :param a_prev: The activations of the previous time-step. (n_a, m)
  :param c_prev: The memory cell of the previous time-step. (n_a, m)

  No longer using a python dict, instead using tensorflow's variable sharing
    wf - weights for the forget gate. (n_a, n_a + n_x)
    bf - bias for the forget gate. (n_a, 1)
    wu - weights for the update gate. (n_a, n_a + n_x)
    bu - bias for the update gate. (n_a, 1)
    wo - weights for the output gate. (n_a, n_a + n_x)
    bo - bias for the output gate. (n_a, 1)
    wc - weights for the c candidate
    bc - bias for the c candidate
    wy - weights for the output (n_y, n_a)
    by - bias for the output. (n_y, 1)
  :return:
    a_next - the activations for this time-step (n_a, m)
    c_next - the memory cell for this time-step (n_a, m)
    y - the output of this time-step. Will be a 3d matrix
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


    #Concatenate the previous activation and the input (for speedup)
    a_x = tf.concat([a_prev, x], axis=0)
    #Calculate the candidate to replace c.
    c_cand = tf.tanh(wc @ a_x + bc)
    #Calculate the update gate.
    u_gate = tf.sigmoid(wu @ a_x + bu)
    #Calculate the forget gate.
    f_gate = tf.sigmoid(wf @ a_x + bf)
    #Calculate the output gate
    o_gate = tf.sigmoid(wo @ a_x + bo)
    #Calulate c, a, and y for this time-step
    c_next = (u_gate * c_cand) + (f_gate * c_prev)
    a_next = o_gate * tf.tanh(c_next)
    y = wy @ a_next + by
    return a_next, c_next, y

def init_lstm(input_size, activation_size, path=None):
  """
  Initializes the weights and biases for the lstm
  :param input_size: the size of the input vectors
  :param activation_size: the size of the activation vector
  :param path: a string that denotes a path to predefined weights
  :return:
  """
  assert type(path) is str or path == None
  with tf.variable_scope('lstm'):
    tf.get_variable('wf', shape=(activation_size, activation_size + input_size), dtype=tf.complex64,
                    initializer=tf.initializers.zeros)
    tf.get_variable('bf', shape=(activation_size, 1), dtype=tf.complex64,
                    initializer=tf.initializers.zeros)
    tf.get_variable('wu', shape=(activation_size, activation_size + input_size), dtype=tf.complex64,
                    initializer=tf.initializers.zeros)
    tf.get_variable('bu', shape=(activation_size, 1), dtype=tf.complex64,
                    initializer=tf.initializers.zeros)
    tf.get_variable('wc', shape=(activation_size, activation_size + input_size), dtype=tf.complex64,
                    initializer=tf.initializers.zeros)
    tf.get_variable('bc', shape=(activation_size, 1), dtype=tf.complex64,
                    initializer=tf.initializers.zeros)
    tf.get_variable('wo', shape=(activation_size, activation_size + input_size), dtype=tf.complex64,
                    initializer=tf.initializers.zeros)
    tf.get_variable('bo', shape=(activation_size, 1), dtype=tf.complex64,
                    initializer=tf.initializers.zeros)
    tf.get_variable('wy', shape=(input_size, activation_size), dtype=tf.complex64,
                    initializer=tf.initializers.zeros)
    tf.get_variable('by', shape=(input_size, 1), dtype=tf.complex64,
                    initializer=tf.initializers.zeros)
    tf.global_variables_initializer()




