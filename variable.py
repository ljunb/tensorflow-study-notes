import tensorflow as tf

# define a variable names value, it's value is 0
value = tf.Variable(0, name = 'counter')
# define constant one
one = tf.constant(1)

# calculate value add one, and then assign new_value to value
new_value = tf.add(value, one)
update = tf.assign(value, new_value)

# must initial all variables if define variable
init = tf.global_variables_initializer()

# with block will close session automatically, just like file operation
with tf.Session() as sess:
    # to active all variables
    sess.run(init)
    for _ in range(3):
        # should call run to handle data every time
        sess.run(update)
        print(sess.run(value))
