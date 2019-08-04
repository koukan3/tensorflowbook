#-*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
'''
@file: 1.3placeholder.py
@description:
placeholder机制：在会话运行时动态提供输入数据。
'''

x1 = tf.placeholder(tf.float32,shape=(2,))
x2 = tf.placeholder(tf.float32,shape=(2,))
result = x1 + x2

with tf.Session() as ss:
    sum = ss.run(result,feed_dict={x1:[3.0,3.0],x2:[4.0,4.0]})
    print(sum)
