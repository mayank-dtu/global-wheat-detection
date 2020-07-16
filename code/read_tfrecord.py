'''
Created on 14-Jul-2020

@author: user
'''
import tensorflow as tf

for example in tf.python_io.tf_record_iterator("/home/user/Mayank/global-wheat-detection/traing.record"):
    print(tf.train.Example.FromString(example))