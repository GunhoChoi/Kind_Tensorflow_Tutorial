import tensorflow as tf
import numpy as np

input_data=np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]],dtype=np.float32)
output_data=np.array([[0,0,1,1]],dtype=np.float32).T

x =tf.placeholder(dtype=np.float32,shape=[4,3])
y_=tf.placeholder(dtype=np.float32,shape=[4,1])

weight=tf.Variable(np.random.random_sample([3,1]),dtype=tf.float32,name='weight')
bias=tf.Variable(np.random.random_sample([1]),dtype=tf.float32,name='bias')

y=tf.matmul(x,weight)+bias

Loss=tf.reduce_mean(tf.square(y-y_))
train=tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(Loss)

feed_dict={x:input_data, y_:output_data}

init=tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for i in range(10000):
		_,l=sess.run([train,Loss],feed_dict=feed_dict)
		if i%100==0:
			print(l)
	print(Loss)
	print(y)
	print(sess.run(Loss,feed_dict=feed_dict))
	print(sess.run(y,feed_dict=feed_dict))