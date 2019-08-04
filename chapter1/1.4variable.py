#-*- coding:utf-8 -*-
import tensorflow as tf
'''
@file: 1.4variable.py
@description:
创建变量
变量初始值
初始化变量
'''
#创建变量：3x2的符合正太分布的矩阵
weight = tf.Variable(tf.random_normal(shape=(3,2),mean=0,stddev=1))
#初始化变量：方式1--全部初始化
#iav = tf.global_variables_initializer()
#创建会话，并作为默认会话
with tf.Session() as ss:
    #ss.run(iav)
    #初始化变量：方式2:---单个初始化
    ss.run(weight.initializer)
    #print(weight.eval())
    print(ss.run(weight))
    #使用其他已经初始化的变量的初始值来创建变量
    weight2 = tf.Variable(weight.initialized_value()*2)
    ss.run(weight2.initializer)
    print(weight2.eval())
