import tensorflow as tf
import numpy
x = numpy.arange(-10, 10, 0.1)
y = -0.1*x**2 + x + numpy.sin(x) + 0.1*numpy.random.randn(len(x))

X = tf.placeholder("float") # create symbolic variables
Y = tf.placeholder("float")

w = tf.Variable(0.0, name="coeff")
b = tf.Variable(0.0, name="offset")
y_model = tf.multiply(X, w) + b

cost = tf.pow(y_model-Y, 2) # use sqr error for cost function

def acost(a):
   return tf.pow(y_model-Y, 2) * tf.pow(tf.sign(y_model-Y) + a, 2)
#here I want to define another cost function, how to pass the parameter the "cost_value" in to this function, cost_value is the output in the session
def bcost(cost_value):
   return tf.pow(y_model-Y, 2) * tf.pow(tf.sign(y_model-Y) + cost_value, 2)

train_op = tf.train.AdamOptimizer().minimize(cost)
train_op2 = tf.train.AdamOptimizer().minimize(acost(-0.5))
train_op3 = tf.train.AdamOptimizer().minimize(bcost(cost_value))
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(100):
    for (xi, yi) in zip(x, y):
    #  sess.run(train_op, feed_dict={X: xi, Y: yi})
    _, cost_value = sess.run([train_op2, cost], feed_dict={X: xi, Y: yi})
    #  _, cost_value = sess.run([train_op3, cost], feed_dict={X: xi, Y: yi})
    # print(tf.float32(sess.run(cost_value)))

print(sess.run(w), sess.run(b))
