import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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

# create data
x_data = np.linspace(-1, 1, 300, dtype = np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise
    
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# hidden layer
l1 = add_layer(xs, 1, 10, activation_function = tf.nn.relu)
# output layer
prediction = add_layer(l1, 10, 1, activation_function = None)

# use GradientDescentOptimizer to optimize the error between real data and output layer
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices = [1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# use with block to manager session automatically
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    # plot the real data
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)
    plt.ion()
    plt.show()
    
    for i in range(2000):
        # traning 2000 times
        sess.run(train_step, feed_dict = {xs: x_data, ys: y_data})
        if i % 50 == 0:
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value = sess.run(prediction, feed_dict = {xs: x_data})
            
            lines = ax.plot(x_data, prediction_value, 'r-', lw = 5)
            plt.pause(1)
