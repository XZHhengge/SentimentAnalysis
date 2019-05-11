# -*- coding:utf-8 -*-
# Author: cmzz
# @Time :2019/5/11
import tensorflow as tf


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # biases =