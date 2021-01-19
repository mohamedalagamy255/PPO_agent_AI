#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# model

import time
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, LSTMCell
import numpy as np
from config import *
from datetime import datetime

logdir      = "/content/drive/My Drive/AI/summary"
weight_dir  = "/content/drive/My Drive/AI/weight/"

#file_writer = tf.summary.create_file_writer(logdir + "/metrics")
#file_writer.set_as_default()

try : 
  data_summary = np.load(os.path.join(logdir , sorted(os.listdir(logdir), key = lambda x : -1 * int(x[8:-4]))[0] ) , allow_pickle = True ).tolist()
  print(data_summary)
  index        = data_summary["step"]
  print(index)
  #index = 10201
except :
  index = 0

if index == 0:
    restore = False
else:
    restore = True
restore_weight_dir = weight_dir + "model_weights"


class CNNModel(Model):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.c1          = Conv2D(32, kernel_size=(8, 8), strides=(4, 4), activation='relu')
        self.c2          = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu')
        self.c3          = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu')
        self.flatten     = Flatten()
        self.d1          = Dense(512, activation="relu")
        self.d2          = Dense(1)  # C
        self.d3          = Dense(a_num, activation='softmax')  # A
        self.total_index = index
        
        self.call(np.random.random((batch_size, IMG_H, IMG_W, k)).astype(np.float32))
        if restore:
            self.load_weights(restore_weight_dir)

    @tf.function
    def call(self, inputs):
        x = inputs / 255.0
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.flatten(x)
        x = self.d1(x)
        a = self.d3(x)
        v = self.d2(x)
        return a, v

    def loss(self, inputs, action_index, adv, targets, ep_old_ap):
        res           = self.call(inputs)
        error         = res[1][:, 0] - targets
        L             = tf.reduce_sum(tf.square(error))

        adv           = tf.dtypes.cast(tf.stop_gradient(adv), tf.float32)
        batch_size    = inputs.shape[0]
        all_act_prob  = res[0]
        selected_prob = tf.reduce_sum(action_index * all_act_prob, axis=1)
        old_prob      = tf.reduce_sum(action_index * ep_old_ap, axis=1)

        r             = selected_prob / (old_prob + 1e-6)

        H             = -tf.reduce_sum(all_act_prob * tf.math.log(all_act_prob + 1e-6))

        Lclip         = tf.reduce_sum(tf.minimum(tf.multiply(r, adv),
                                                 tf.multiply(tf.clip_by_value(r,1 - clip_epsilon,1 + clip_epsilon),adv)))

        return -(Lclip - VFcoeff * L + beta * H) / batch_size, Lclip, H, L 

    def total_grad(self, ep_stack_obs, ep_as, adv, realv, ep_old_ap):
        with tf.GradientTape() as tape:
            loss_value, Lclip, H, L = self.loss(ep_stack_obs, ep_as, adv, realv, ep_old_ap)
            self.total_index       += 1

            if self.total_index % recode_span == 1:
                #self.record('total loss', loss_value * len(ep_as))
                #self.record('Lclip', Lclip)
                #self.record('H', H)
                #self.record('c loss', L)
                data_summary               = {}
                data_summary['total loss'] = loss_value * len(ep_as)
                data_summary['Lclip']      = Lclip
                data_summary['H']          = H
                data_summary['c loss']     = L
                data_summary["step"]       = self.total_index
                np.save(logdir + "/" + "summary_{}".format(self.total_index), np.array(data_summary))


            if self.total_index % save_span == 0:
                self.save_weights(weight_dir + "model_weights", save_format='tf')
        return tape.gradient(loss_value, self.trainable_weights), loss_value

    def record(self, name, data, step=None):
        if not step:
            step = self.total_index
        tf.summary.scalar(name, data=data, step=step)


