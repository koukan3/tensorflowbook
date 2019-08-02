import tensorflow as tf

'''
tensor张量只是保存了运算结果的属性.
例如：Tensor("Add:0", shape=(2, 3), dtype=int32)，
其中包括：
操作op----add，是节点的第0个输出。
维度shape
数据类型dtype
'''

c1 = tf.constant(2,shape=(2,3))
c2 = tf.constant(10)
add_result = tf.add(c1,c2)
print(add_result) #Tensor("Add:0", shape=(2, 3), dtype=int32)

ss = tf.Session()
print(ss.run(add_result))