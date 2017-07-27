import tensorflow as tf
import numpy as np

'''
    To create our own Neural Networks, we could define a function to add layer.
    Neural layer should have weights, biases and activation function.
'''
def add_layer(inputs, in_size, out_size, activation_function=None):
    # init weights and biases
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # the function expression: y = x * weights + biases
    wx_plus_b = tf.matmul(inputs, weights) + biases
    # handle our own activation function if activation_function is None
    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activation_function(wx_plus_b)
    return outputs


    
