import tensorflow as tf

'''
tensorflow计算图
'''

g1 = tf.Graph()
with g1.as_default():
    v1 = tf.get_variable("v1",[2],initializer=tf.ones_initializer())
    v2 = tf.get_variable("v2",[2],initializer=tf.zeros_initializer())

g2 = tf.Graph()
with g2.as_default():
    v1 = tf.get_variable("v1",[2],initializer=tf.zeros_initializer())
    v2 = tf.get_variable("v2",[2],initializer=tf.ones_initializer())

#将计算图交由会话session执行
#with/as:环境上下文管理器，是异常相关的语句，可解决资源泄露问题。
with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run() #初始化计算图中所有变量
    with tf.variable_scope("vs",reuse=tf.AUTO_REUSE):#控制变量空间
        #在变量空间内创建的变量名称都会带上这个变量空间名称作为前缀,例如：vs/v1:0
        print(sess.run(tf.get_variable("v1")))
        print(sess.run(tf.get_variable("v2")))

with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("",reuse=tf.AUTO_REUSE):
        print(sess.run(tf.get_variable("v1")))
        print(sess.run(tf.get_variable("v2")))