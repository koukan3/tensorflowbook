#-*- coding:utf-8 -*-
import tensorflow as tf
'''
@file: 1.6variable.py
@description:
变量赋值，改变维度
'''

w1 = tf.Variable(tf.random_normal(shape=(2,3)))
w2 = tf.Variable(tf.random_normal(shape=(3,1)))
#将w2的值赋给w1
result = tf.assign(w1,w2,validate_shape=False)
inits = tf.global_variables_initializer()
with tf.Session() as ss:
    ss.run(inits)
    print(ss.run(w1))
    print(ss.run(w2))
    print(ss.run(result))