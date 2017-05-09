5. should we modify for test time?
9. note that using tf.nn.moments


now try with bigger longer learning


init weights like in

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)