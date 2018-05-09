# ----------------------------------------------------------
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
# -----------------------------------------------------------

import tensorflow as tf

from deprecated.initializers import complex_random_uniform


def init_lstm(input_size, activation_size, path=None):
  """
  Initializes the weights and biases for the lstm
  :param input_size: the size of the input vectors
  :param activation_size: the size of the activation vector
  :param path: a string that denotes a path to predefined weights
  :return:
  """
  if path is None:
    with tf.variable_scope('lstm', dtype=tf.complex64, initializer=complex_random_uniform):
      tf.get_variable('wf', shape=(activation_size, activation_size + input_size))
      tf.get_variable('bf', shape=(activation_size, 1))
      tf.get_variable('wu', shape=(activation_size, activation_size + input_size))
      tf.get_variable('bu', shape=(activation_size, 1))
      tf.get_variable('wc', shape=(activation_size, activation_size + input_size))
      tf.get_variable('bc', shape=(activation_size, 1))
      tf.get_variable('wo', shape=(activation_size, activation_size + input_size))
      tf.get_variable('bo', shape=(activation_size, 1))
      tf.get_variable('wy', shape=(input_size, activation_size))
      tf.get_variable('by', shape=(input_size, 1))


def complex_lstm_cell(x, a_prev, c_prev, timestep):
  """

  :param x: Input vector at the current time-step. (n_x, m)
  :param a_prev: The activations of the previous time-step. (n_a, m)
  :param c_prev: The memory cell of the previous time-step. (n_a, m)
  :param timestep: The timestep that the cell is being executed in

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

    # Concatenate the previous activation and the input (for speedup)
    a_x = tf.concat([a_prev, x], axis=0)
    # Calculate the candidate to replace c.
    c_cand = tf.tanh(wc @ a_x + bc)
    # Calculate the update gate.
    u_gate = tf.sigmoid(wu @ a_x + bu)
    # Calculate the forget gate.
    f_gate = tf.sigmoid(wf @ a_x + bf)
    # Calculate the output gate
    o_gate = tf.sigmoid(wo @ a_x + bo)
    # Calulate c, a, and y for this time-step
    c_next = (u_gate * c_cand) + (f_gate * c_prev)
    a_next = o_gate * tf.tanh(c_next)
    y = wy @ a_next + by

    # We need to pack the intermediate values into a collection for use in backprop
    # The order of package is a_x, c_prev, c_cand, u_gate, f_gate, o_gate, c_next, a_next
    with tf.variable_scope(str(timestep)):
      cache = 'cache'
      tf.add_to_collection(cache, a_x)
      tf.add_to_collection(cache, c_prev)
      tf.add_to_collection(cache, c_cand)
      tf.add_to_collection(cache, u_gate)
      tf.add_to_collection(cache, f_gate)
      tf.add_to_collection(cache, o_gate)
      tf.add_to_collection(cache, c_next)
      tf.add_to_collection(cache, a_next)
    return a_next, c_next, y


def complex_lstm_cell_back(dy, da_next, dc_next, timestep):
  """
  :param dy: The gradient coming down from an upper layer (either cost or another lstm)
  :param da_next: The gradient of the activation vector from the next timestep
  :param dc_next: The gradient of the memory cell from the next timestep
  :param timestep: The timestep that this cell is in
  :return:
    da_pass: The gradient of the activation vector to be sent to the previous timestep
    dx_pass: The gradient of the input to be sent to any layers underneath this lstm
    dc_pass: The gradient of the memory cell to be sent to the previous timestep
    Note: Once this has finished across the training set, it will have created collections with the gradients
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
    [a_x, c_prev, c_cand, u_gate, f_gate, o_gate, c, a] = tf.get_collection('cache', 'lstm/' + str(timestep))

    da = tf.transpose(wy) @ dy + da_next

    dwy = dy @ tf.transpose(a)
    dby = dy

    dc = da * o_gate * (1 - tf.tanh(c) ** 2) + dc_next

    do_gate = da * tf.tanh(c) * o_gate * (1 - o_gate)
    df_gate = dc * c_prev * f_gate * (1 - f_gate)
    du_gate = c_cand * dc * u_gate * (1 - u_gate)

    dc_cand = u_gate * (1 - tf.tanh(c_cand) ** 2) * dc

    dwo = du_gate @ tf.transpose(a_x)
    dbo = do_gate
    dxo = tf.transpose(wo) @ du_gate

    dwf = df_gate @ tf.transpose(a_x)
    dbf = df_gate
    dxf = tf.transpose(wf) @ df_gate

    dwu = du_gate @ tf.transpose(a_x)
    dbu = du_gate
    dxu = tf.transpose(wu) @ du_gate

    dwc = dc_cand @ tf.transpose(a_x)
    dbc = dc_cand
    dxc = tf.transpose(wc) @ dc_cand

    dx = dxo + dxf + dxu + dxc

    input_size = tf.shape(dc)[0]
    # The values we will be passing back to the previous timestep
    da_pass = dx[:input_size, :]
    dx_pass = dx[input_size:, :]
    dc_pass = dc * f_gate
    # Lets put the gradients into collections
    tf.add_to_collection('dwy', dwy)
    tf.add_to_collection('dby', dby)
    tf.add_to_collection('dwo', dwo)
    tf.add_to_collection('dbo', dbo)
    tf.add_to_collection('dwf', dwf)
    tf.add_to_collection('dbf', dbf)
    tf.add_to_collection('dwu', dwu)
    tf.add_to_collection('dbu', dbu)
    tf.add_to_collection('dwc', dwc)
    tf.add_to_collection('dbc', dbc)
    return da_pass, dx_pass, dc_pass


def complex_lstm_forward_training(X, a0, c0):
  """
  This method allows for you to execute the lstm
  :param X: input Vectors should be of size (stfsBins, Frames)
  :param a0: initial state activations
  :param c0: initial state memory cell
  :return: The outputs of the model
  """

  outputs = tf.TensorArray(tf.complex64, size=2583)
  i = tf.constant(0)

  def body(i, a, c, outputs):
    a_next, c_next, out = complex_lstm_cell(X[:, i:i + 1], a, c)
    outputs = outputs.write(i, out)
    return i + 1, a_next, c_next, outputs

  def cond(i, a, c, outputs):
    return i < tf.shape(X)[1]

  __, __, __, outputs = tf.while_loop(cond, body, (i, a0, c0, outputs))
  # Convert the outputs into a tensor
  return tf.transpose(tf.squeeze(outputs.stack()))

def complex_lstm_back_training(loss):
  None
