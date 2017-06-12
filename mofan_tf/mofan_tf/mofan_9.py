import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7],input2:[2]}))

"""
为什么需要激励函数：
线性方程： y = wx
加上激励函数：y = AF(wx)  /relu sigmoid tanh 掰完利器
CNN选relu/RNN选relu或者tanh
"""