#-*- coding:utf-8 -*-
import tensorflow as tf
'''
@file: 1.5variable.py
@description:
变量实现矩阵相乘操作
'''
#构造矩阵
x_inputs = tf.constant([[1,2]],dtype=tf.float32)
weight1 = tf.Variable(tf.random_normal((2,3),mean=0,stddev=1,seed=1))
weight2 = tf.Variable(tf.random_normal((3,1),mean=0,stddev=1,seed=1))
bias = tf.ones(shape=(1,3))
#矩阵相乘
output1 = tf.matmul(x_inputs,weight1) + bias
result = tf.matmul(output1,weight2)
#创建session，作为默认session
ss = tf.InteractiveSession()
#初始化变量
inits = tf.global_variables_initializer()
ss.run(inits)
print(ss.run(result))




