import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, datasets, optimizers, metrics, layers, callbacks,losses


class LinfPGDAttack:
    def __init__(self,model,epsilon,alpha,k,random_start=False,batch_size=2):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.t = k
        self.randstart = random_start
        self.batch = batch_size

    def perturb_po(self,x_positive_nat,x_negative_nat,y):
        if self.epsilon == 0:
            return x_positive_nat
        all_num = x_positive_nat.shape[0]
        num = all_num // self.batch
        if self.randstart:
            x_ad = x_positive_nat + np.random.uniform(-self.epsilon,self.epsilon,x_positive_nat.shape)
        else:
            x_ad = x_positive_nat

        x_AD = np.zeros(x_positive_nat.shape,dtype=np.float64)
        x_positive_nat = np.expand_dims(x_positive_nat,axis=1)

        for n in range(num):
            x_positive_temp = x_ad[int(n*self.batch):int((n+1)*self.batch),:]
            x_negative_temp = x_negative_nat[int(n*self.batch):int((n+1)*self.batch),:]
            x_positive_temp = tf.expand_dims(x_positive_temp, axis=1)
            x_negative_temp = tf.expand_dims(x_negative_temp, axis=1)

            mask = y[int(n*self.batch):int((n+1)*self.batch)]
            mask = tf.one_hot(mask,depth=2)
            x_ne = tf.concat([x_positive_temp, x_negative_temp], axis=1)
            for i in range(int(self.t)):
                with tf.GradientTape() as tape:


                    tape.watch(x_ne)

                    y_pre = self.model(x_ne)
                    y_pre = tf.squeeze(y_pre)
                    loss = tf.keras.losses.binary_crossentropy(y_true=mask,y_pred=y_pre)
                    dl_x_temp = tape.gradient(loss,x_ne)[0]


                x_positive_temp += self.alpha * np.sign(dl_x_temp[0,:])

                x_positive_temp = np.clip(x_positive_temp,x_positive_nat[int(n*self.batch):int((n+1)*self.batch),:] - self.epsilon,x_positive_nat[int(n*self.batch):int((n+1)*self.batch),:] + self.epsilon)
                x_positive_temp = np.clip(x_positive_temp, 0, 1)

            x_positive_temp = tf.squeeze(x_positive_temp)
            x_AD[int(n*self.batch):int((n+1)*self.batch),:] = x_positive_temp

        x_positive_temp = x_ad[int(num * self.batch):, :]
        x_negative_temp = x_negative_nat[int(num * self.batch):, :]
        x_positive_temp = tf.expand_dims(x_positive_temp,axis=1)
        x_negative_temp = tf.expand_dims(x_negative_temp,axis=1)
        x_ne = tf.concat([x_positive_temp,x_negative_temp],axis=1)
        mask = y[int(num * self.batch):]
        mask = tf.one_hot(mask,depth=2)
        for i in range(int(self.t)):
            with tf.GradientTape() as tape:

                tape.watch(x_ne)
                y_pre = self.model(x_ne)
                y_pre = tf.squeeze(y_pre)
                loss = tf.keras.losses.binary_crossentropy(mask, y_pre)
                dl_x_temp = tape.gradient(loss, x_ne)[0]

            x_positive_temp += self.alpha * np.sign(dl_x_temp[0,:])

            x_positive_temp = np.clip(x_positive_temp, x_positive_nat[int(num * self.batch):, :] - self.epsilon,
                             x_positive_nat[int(num * self.batch):, :] + self.epsilon)
            x_positive_temp = np.clip(x_positive_temp, 0, 1)

        x_positive_temp = tf.squeeze(x_positive_temp)
        x_AD[int(num)*self.batch:,:] = x_positive_temp

        return  x_AD

    def perturb_ne(self,x_positive_nat,x_negative_nat,y):
        if self.epsilon == 0:
            return x_negative_nat
        all_num = x_negative_nat.shape[0]
        num = all_num // self.batch
        if self.randstart:
            x_ad = x_negative_nat + np.random.uniform(-self.epsilon,self.epsilon,x_negative_nat.shape)
        else:
            x_ad = x_negative_nat

        x_AD = np.zeros(x_negative_nat.shape,dtype=np.float64)
        x_negative_nat = tf.expand_dims(x_negative_nat,axis=1)
        for n in range(num):
            x_negative_temp = x_ad[int(n*self.batch):int((n+1)*self.batch),:]
            x_positive_temp = x_positive_nat[int(n*self.batch):int((n+1)*self.batch),:]
            x_negative_temp = tf.expand_dims(x_negative_temp,axis=1)
            x_positive_temp = tf.expand_dims(x_positive_temp,axis=1)
            x_ne = tf.concat([x_positive_temp,x_negative_temp],axis=1)
            mask = y[int(n*self.batch):int((n+1)*self.batch)]
            mask = tf.one_hot(mask,depth=2)
            for i in range(int(self.t)):
                with tf.GradientTape() as tape:

                    tape.watch(x_ne)
                    y_pre = self.model(x_ne)

                    y_pre = tf.squeeze(y_pre)
                    loss = tf.keras.losses.binary_crossentropy(y_true=mask,y_pred=y_pre)
                    dl_x_temp = tape.gradient(loss,x_ne)[0]


                x_negative_temp += self.alpha * np.sign(dl_x_temp[1,:])

                x_negative_temp = np.clip(x_negative_temp,x_negative_nat[int(n*self.batch):int((n+1)*self.batch),:] - self.epsilon,x_negative_nat[int(n*self.batch):int((n+1)*self.batch),:] + self.epsilon)
                x_negative_temp = np.clip(x_negative_temp, 0, 1)

            x_negative_temp = tf.squeeze(x_negative_temp)
            x_AD[int(n*self.batch):int((n+1)*self.batch),:] = x_negative_temp

        x_negative_temp = x_ad[int(num * self.batch):, :]
        x_positive_temp = x_positive_nat[int(num * self.batch):, :]
        x_negative_temp = tf.expand_dims(x_negative_temp,axis=1)
        x_positive_temp = tf.expand_dims(x_positive_temp,axis=1)
        x_ne = tf.concat([x_positive_temp,x_negative_temp],axis=1)
        mask = y[int(num * self.batch):]
        mask = tf.one_hot(mask,depth=2)
        for i in range(int(self.t)):
            with tf.GradientTape() as tape:
                tape.watch(x_ne)
                y_pre = self.model(x_ne)
                y_pre = tf.squeeze(y_pre)
                loss = tf.keras.losses.binary_crossentropy(mask, y_pre)
                dl_x_temp = tape.gradient(loss, x_ne)[0]

            x_negative_temp += self.alpha * np.sign(dl_x_temp[1,:])
            x_negative_temp = np.clip(x_negative_temp, x_negative_nat[int(num * self.batch):, :] - self.epsilon,
                             x_negative_nat[int(num * self.batch):, :] + self.epsilon)
            x_negative_temp = np.clip(x_negative_temp, 0, 1)

        x_negative_temp = tf.squeeze(x_negative_temp)
        x_AD[int(num)*self.batch:,:] = x_negative_temp

        return  x_AD


