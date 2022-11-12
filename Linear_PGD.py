import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, datasets, optimizers, metrics, layers, callbacks,losses


class LinfPGDAttack:  # Linear PGD Algorithm
    def __init__(self,model,epsilon,alpha,k,random_start=False,batch_size=2):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.t = k
        self.randstart = random_start
        self.batch = batch_size

    def perturb_po(self,x_positive_nat,x_negative_nat,y):  # perturbation positive data
        if self.epsilon == 0:
            return x_positive_nat
        all_num = x_positive_nat.shape[0]
        num = all_num // self.batch
        if self.randstart:
            x_ad = x_positive_nat + np.random.uniform(-self.epsilon,self.epsilon,x_positive_nat.shape)
        else:
            x_ad = x_positive_nat

        x_AD = np.zeros(x_positive_nat.shape,dtype=np.float64)
        for n in range(num):
            x_positive_temp = x_ad[int(n*self.batch):int((n+1)*self.batch),:]
            x_negative_temp = x_negative_nat[int(n*self.batch):int((n+1)*self.batch),:]
            mask = y[int(n*self.batch):int((n+1)*self.batch)]
            for i in range(int(self.t)):
                with tf.GradientTape() as tape:
                    x_positive_temp = tf.constant(x_positive_temp)
                    x_negative_temp = tf.constant(x_negative_temp)
                    tape.watch(x_positive_temp)
                    x_ne = x_positive_temp-x_negative_temp

                    y_pre = self.model(x_ne)
                    loss = tf.keras.losses.hinge(y_true=mask,y_pred=y_pre)
                    dl_x_temp = tape.gradient(loss,x_positive_temp)[0]


                x_positive_temp += self.alpha * np.sign(dl_x_temp)
                x_positive_temp = np.clip(x_positive_temp,x_positive_nat[int(n*self.batch):int((n+1)*self.batch),:] - self.epsilon,x_positive_nat[int(n*self.batch):int((n+1)*self.batch),:] + self.epsilon)
                x_positive_temp = np.clip(x_positive_temp, 0, 1)

            x_AD[int(n*self.batch):int((n+1)*self.batch),:] = x_positive_temp

        x_positive_temp = x_ad[int(num * self.batch):, :]
        x_negative_temp = x_negative_nat[int(num * self.batch):, :]
        mask = y[int(num * self.batch):]
        for i in range(int(self.t)):
            with tf.GradientTape() as tape:
                x_positive_temp = tf.constant(x_positive_temp)
                tape.watch(x_positive_temp)
                y_pre = self.model(x_positive_temp-x_negative_temp)
                loss = tf.keras.losses.hinge(mask, y_pre)
                dl_x_temp = tape.gradient(loss, x_positive_temp)[0]

            x_positive_temp += self.alpha * np.sign(dl_x_temp)
            x_positive_temp = np.clip(x_positive_temp, x_positive_nat[int(num * self.batch):, :] - self.epsilon,
                             x_positive_nat[int(num * self.batch):, :] + self.epsilon)
            x_positive_temp = np.clip(x_positive_temp, 0, 1)


        x_AD[int(num)*self.batch:,:] = x_positive_temp

        return  x_AD

    def perturb_ne(self,x_positive_nat,x_negative_nat,y):  # perturbation negative data
        if self.epsilon == 0:
            return x_negative_nat
        all_num = x_negative_nat.shape[0]
        num = all_num // self.batch
        if self.randstart:
            x_ad = x_negative_nat + np.random.uniform(-self.epsilon,self.epsilon,x_negative_nat.shape)
        else:
            x_ad = x_negative_nat

        x_AD = np.zeros(x_negative_nat.shape,dtype=np.float64)
        for n in range(num):
            x_negative_temp = x_ad[int(n*self.batch):int((n+1)*self.batch),:]
            x_positive_temp = x_positive_nat[int(n*self.batch):int((n+1)*self.batch),:]
            mask = y[int(n*self.batch):int((n+1)*self.batch)]
            for i in range(int(self.t)):
                with tf.GradientTape() as tape:
                    x_positive_temp = tf.constant(x_positive_temp)
                    x_negative_temp = tf.constant(x_negative_temp)
                    tape.watch(x_negative_temp)
                    x_ne = x_positive_temp - x_negative_temp
                    y_pre = self.model(x_ne)
                    loss = tf.keras.losses.hinge(y_true=mask,y_pred=y_pre)
                    dl_x_temp = tape.gradient(loss,x_negative_temp)[0]


                x_negative_temp += self.alpha * np.sign(dl_x_temp)
                x_negative_temp = np.clip(x_negative_temp,x_negative_nat[int(n*self.batch):int((n+1)*self.batch),:] - self.epsilon,x_negative_nat[int(n*self.batch):int((n+1)*self.batch),:] + self.epsilon)
                x_negative_temp = np.clip(x_negative_temp, 0, 1)

            x_AD[int(n*self.batch):int((n+1)*self.batch),:] = x_negative_temp

        x_negative_temp = x_ad[int(num * self.batch):, :]
        x_positive_temp = x_positive_nat[int(num * self.batch):, :]
        mask = y[int(num * self.batch):]
        for i in range(int(self.t)):
            with tf.GradientTape() as tape:
                x_negative_temp = tf.constant(x_negative_temp)
                x_positive_temp = tf.constant(x_positive_temp)
                tape.watch(x_negative_temp)
                y_pre = self.model(x_positive_temp-x_negative_temp)
                loss = tf.keras.losses.hinge(mask, y_pre)
                dl_x_temp = tape.gradient(loss, x_negative_temp)[0]

            x_negative_temp += self.alpha * np.sign(dl_x_temp)
            x_negative_temp = np.clip(x_negative_temp, x_negative_nat[int(num * self.batch):, :] - self.epsilon,
                             x_negative_nat[int(num * self.batch):, :] + self.epsilon)
            x_negative_temp = np.clip(x_negative_temp, 0, 1)


        x_AD[int(num)*self.batch:,:] = x_negative_temp

        return  x_AD


