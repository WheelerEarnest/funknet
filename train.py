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

from data_processing import *
from lstm import *
import tensorflow as tf


def cost(training_data, model_output):
  """
  In this method, we are looping through the TensorArray, which contains the training examples
  :param training_data: TensorArray containing the input data used for this batch
  :param model_output: TensorArray containing the output of the model
  :return: Returns the cost for these inputs
  """

  cost = tf.constant(0.0)
  i = tf.constant(0)

  def body(i, x, y_hat, J):
    print(tf.shape(loss(x.read(i), y_hat.read(i))))
    J = J + loss(x.read(i), y_hat.read(i))
    J.set_shape(cost.get_shape())
    return i + 1, x, y_hat, J

  def cond(i, x, y_hat, J):
    return i < x.size()

  __, __, __, J = tf.while_loop(cond, body, (i, training_data, model_output, cost))

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
  y_hat = y_hat[:, :-1]
  x = x[:, 1:]
  diff = tf.abs(y_hat - x)
  norm = tf.norm(diff, ord=2, axis=1)
  return tf.reduce_sum(norm)


def feed_data(songs, activation_size):
  """

  :param songs:TensorArray of all the decomposed audio
  :param activation_size: size of the activation vector
  :return: returns the a TensorArray containing the outputs of the net
  """
  outputs = tf.TensorArray(size=songs.size(), dtype=tf.complex64)

  def body(i, outputs):
    out = complex_lstm_forward_training(songs.read(i), tf.zeros((activation_size, 1), dtype=tf.complex64),
                                  tf.zeros((activation_size, 1), dtype=tf.complex64))
    ouputs = outputs.write(i, out)
    return i+1, outputs

  def cond(i, outputs):
    return i < outputs.size()

  __, outputs = tf.while_loop(cond, body, (0, outputs))
  return outputs


def train(iterations):
  frame_length = 1024
  frame_step = 512
  frame_size = frame_length // 2 + 1

  with tf.Session() as sess:
    optimizer = tf.train.AdamOptimizer(0.01)
    init_lstm(frame_size, frame_size)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    sound = np.load('processed/0.npy')
    print(np.shape(sound))
    # outputs = feed_data(trans_songs, frame_size)
    X = tf.placeholder(tf.complex64, shape=np.shape(sound))
    out = complex_lstm_forward_training(X, tf.zeros((frame_size, 1), dtype=tf.complex64),
                                        tf.zeros((frame_size, 1), dtype=tf.complex64))
    J = loss(X, out)
    writer = tf.summary.FileWriter('graph/', sess.graph)
    train_step = optimizer.minimize(J)
    for i in range(iterations):
      sess.run(train_step, feed_dict={X: sound})
      if i % 20 == 0:
        print("iteration: " + str(i))
        print("cost: " + str(J))
        saver.save(sess, 'models/lstm-', i)
    writer.close()


train(1000)




