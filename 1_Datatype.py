# -*- coding: utf8 -*-
import tensorflow as tf

# Constant
c1=tf.constant(1,dtype=tf.float32,name='const_1')
c2=tf.constant(2,dtype=tf.float32,name='const_2')
c3=c1+c2
print(c1,c2,c3)

with tf.Session() as sess:
    print(sess.run(c3))

# Variable
v1=tf.Variable(0,dtype=tf.float32,name='var_1')
v2=tf.Variable(0,dtype=tf.float32,name='var_2')
v1.assign(1)
v2.assign(2)
v3=v1+v2
print(v1,v2,v3)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(v3))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(v1.assign(1))
    sess.run(v2.assign(2))
    print(sess.run(v3))

# placeholder
p1=tf.placeholder(shape=[2,2],dtype=tf.float32,name='place_1')
p2=tf.placeholder(shape=[2,2],dtype=tf.float32,name='place_2')
p3=p1+p2
print(p1,p2,p3)

feed_dict={p1:[[1,2],[3,4]], p2:[[5,6],[7,8]]}
with tf.Session() as sess:
   print(sess.run(p3,feed_dict=feed_dict))
