import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("./data/",one_hot=True)

X_train=tf.placeholder(tf.float32,shape=[None,784])
W=tf.Variable(tf.random_normal([784,10]),name='weights')
b=tf.Variable(tf.random_normal([10]),name='bias')
y_hypo=tf.matmul(X_train,W)+b
y_label=tf.placeholder(tf.float32,shape=[None,10])

cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_label,logits=y_hypo))
train=tf.train.AdamOptimizer(learning_rate=0.01).minimize(cross_entropy)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(50000):
    x_train, y_train = mnist.train.next_batch(125)
    sess.run(train,feed_dict={X_train:x_train,y_label:y_train})
    if i%100==0:
        cross_ = sess.run(cross_entropy, feed_dict={X_train: x_train, y_label: y_train})
        print("Step: ",i+1," cross_entropy: ",cross_,'\n')

score=tf.equal(tf.argmax(y_hypo,1),tf.argmax(y_label,1))
accuracy=tf.reduce_mean(tf.cast(score,tf.float32))
result=sess.run(accuracy,feed_dict={X_train:mnist.test.images,y_label:mnist.test.labels})
print(result)
