#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# brain
import numpy as np
import numba
import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf
import tensorflow.keras.optimizers as optim
from config import *
from model import CNNModel, RNNModel


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, init_lr):
        super(CustomSchedule, self).__init__()
        self.last_lr            = init_lr
        self.max_learning_times = max_learning_times * epochs

    def __call__(self, step):
        # step start from 0
        # every time call , step automatic += 1
        temp_lr      = lr * ((self.max_learning_times - step) / self.max_learning_times)
        self.last_lr = temp_lr
        return temp_lr

    def get_config(self):
        return self.last_lr


class ACBrain():
    def __init__(self, talker):
        super(ACBrain, self).__init__()
        
        self.model                    = CNNModel()
        self.talker                   = talker
        self.i                        = 1
        self.optimizer                = optim.Adam(learning_rate = CustomSchedule(lr))
        self.states_list              = self.talker.states_list
        self.one_episode_reward_index = 0

    def cnn_forward_calc(self, state):
        a_prob, v = self.model(state)
        return a_prob.numpy(), v.numpy()  # [(*,a_num) , (*,1)]  np.array


    def run(self):
        print("brain" + "      ", os.getpid())

        total_obs     = np.zeros((batch_size, process_num, IMG_H, IMG_W, k), dtype=np.float32)
        total_v       = np.zeros((batch_size + 1, process_num), dtype=np.float32)
        total_as      = np.zeros((batch_size, process_num), dtype=np.int32)
        total_rs      = np.zeros((batch_size, process_num), dtype=np.float32)
        total_is_done = np.zeros((batch_size, process_num), dtype=np.float32)
        total_old_ap  = np.zeros((batch_size, process_num, a_num), dtype=np.float32)
        temp_obs      = np.zeros((process_num, IMG_H, IMG_W, k), dtype=np.float32)
        while 1:
            for i in range(batch_size):
                for j in range(process_num):
                    child_id, data              = self.talker.recv()
                    temp_obs[child_id, :, :, :] = np.array(data[0], dtype=np.float32)
                total_obs[i, :, :, :, :] = temp_obs
                a_prob, v                = self.cnn_forward_calc(temp_obs)
                for child_id in range(process_num):
                    self.talker.send(
                        a_prob[child_id],
                        child_id
                    )

                v.resize((process_num,))
                total_v[i, :]         = v
                total_old_ap[i, :, :] = a_prob

            for j in range(process_num):
                child_id, data              = self.talker.recv()
                temp_obs[child_id, :, :, :] = np.array(data[0], dtype=np.float32)

            a_prob, v = self.cnn_forward_calc(temp_obs)
            for child_id in range(process_num):
                self.talker.send(
                    a_prob[child_id],
                    child_id
                )
            v.resize((process_num,))
            total_v[-1, :] = v

            for j in range(process_num):
                child_id, data             = self.talker.recv()
                # data
                # [
                # self.send_as,
                # self.send_rs,
                # self.send_is_done,
                # self.episode_reward
                # ]
                total_as[:, child_id]      = data[0]
                total_rs[:, child_id]      = data[1]
                total_is_done[:, child_id] = data[2]
                for one_episode_reward in data[3]:  # use tensorflow recode reward in one episode
                    self.model.record(name = 'one_episode_reward', data = one_episode_reward,
                                      step = self.one_episode_reward_index)
                    self.one_episode_reward_index += 1

            total_realv, total_adv = self.calc_realv_and_adv_GAE(total_v, total_rs, total_is_done)

            total_obs.resize((process_num * batch_size, IMG_H, IMG_W, k))
            total_as.resize((process_num * batch_size,))
            total_old_ap.resize((process_num * batch_size, a_num))
            total_adv.resize((process_num * batch_size,))
            total_realv = total_realv.reshape((process_num * batch_size,))
      
            self.cnn_learn(
                total_obs,
                tf.one_hot(total_as, depth=a_num).numpy(),
                total_old_ap,
                total_adv,
                total_realv
                )

            if self.i == max_learning_times:
                #from model import weight_dir
                self.model.save_weights(weight_dir + str(self.model.total_index), save_format='tf')
                print("learning done")
                return 0

            self.i += 1
            print("-------------------------")
            print(self.i)

            for child_id in range(process_num):  # tell agents that can start act with env
                self.states_list[child_id] = 0
                self.talker.send("ok", child_id)

            total_obs.resize((batch_size, process_num, IMG_H, IMG_W, k))
            total_as.resize((batch_size, process_num,))
            total_old_ap.resize((batch_size, process_num, a_num))
            total_adv.resize((batch_size, process_num,))

    def cnn_learn(self, total_obs, total_as, total_old_ap, total_adv, total_real_v):

        for _ in range(epochs):
            sample_index = np.random.choice(total_as.shape[0], size=learning_batch)
            grads, loss = self.model.total_grad(total_obs[sample_index],
                                                total_as[sample_index],
                                                total_adv[sample_index],
                                                total_real_v[sample_index],
                                                total_old_ap[sample_index])
            grads, grad_norm = tf.clip_by_global_norm(grads, 0.5)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        

    @staticmethod
    @numba.jit(nopython=True)
    def calc_realv_and_adv(v, r, done):
        length = r.shape[0]
        num = r.shape[1]

        realv = np.zeros((length + 1, num), dtype=np.float32)
        adv = np.zeros((length, num), dtype=np.float32)

        realv[-1, :] = v[-1, :] * (1 - done[-1, :])

        for t in range(length - 1, -1, -1):
            realv[t, :] = realv[t + 1, :] * gamma * (1 - done[t, :]) + r[t, :]
            adv[t, :] = realv[t, :] - v[t, :]

        return realv[:-1, :], adv  # end_v dont need

    @staticmethod
    @numba.jit(nopython=True)
    def calc_realv_and_adv_GAE(v, r, done):
        length = r.shape[0]
        num    = r.shape[1]
        adv    = np.zeros((length + 1, num), dtype=np.float32)

        for t in range(length - 1, -1, -1):
            delta     = r[t, :] + v[t + 1, :] * gamma * (1 - done[t, :]) - v[t, :]
            adv[t, :] = delta + gamma * 0.95 * adv[t + 1, :] * (1 - done[t, :])

        adv   = adv[:-1, :]
        realv = adv + v[:-1, :]

        return realv, adv

