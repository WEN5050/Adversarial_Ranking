import tensorflow as tf
import numpy as np

from tensorflow.keras import Model, datasets, optimizers, metrics, layers, callbacks,losses
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from  tensorflow.keras.metrics import AUC

# Experiment related to the magnitude of weights
# To be consistent with the theorem, our models with L_2,1 regularizer
# Regulaization coefficient is lamda

class Network(Model):
    def __init__(self,dimension,lamd = 0):
        super(Network,self).__init__()
        self.lamd = lamd
        self.input_layer = layers.Input((2,dimension))
        self.full1 = layers.Dense(512,activation='relu',kernel_regularizer=tf.keras.regularizers.l1_l2(self.lamd))

        self.full3 = layers.Dense(256,activation='relu')
        self.full4 = layers.Dense(64,activation='relu')
        self.full5 = layers.Dense(1,tf.nn.sigmoid)

    def call(self, inputs, training=None, mask=None):
        x_full1 =self.full1(inputs)

        x_full3 = self.full3(x_full1)
        x_full4 = self.full4(x_full3)
        x_full5 = self.full5(x_full4)
        # out = self.full5(x_full4)
        return x_full5

    def myloss(self,y,y_pred):
        y_pred = tf.squeeze(y_pred)

        loss = tf.keras.losses.binary_crossentropy(y,y_pred)
        return loss





auc = []
lamd = [0.00,0.002,0.004,0.006,0.008,0.01,0.012,0.014,0.016,0.018,0.02] # Regularization coefficient

for lamd_index in range(len(lamd)):
    print(lamd_index)
    lam = lamd[lamd_index]
    d = 3072 # dimension
    optimizer = optimizers.Adam(lr=1e-4)
    x_po, y_po = np.load('positivedata/cifar_trainx.npy'), np.load('positivedata/cifar_trainy.npy')

    x_ne, y_ne = np.load('negativedata/cifar_trainx.npy'), np.load('negativedata/cifar_trainy.npy')
    x_val_po, y_val_po = np.load('positivedata/cifar_valx.npy'), np.load('positivedata/cifar_valy.npy')
    x_val_ne, y_val_ne = np.load('negativedata/cifar_valx.npy'), np.load('negativedata/cifar_valy.npy')
    x_po, x_ne = tf.reshape(tf.convert_to_tensor(x_po,dtype=tf.float64), [-1, d])/ 255., tf.reshape(tf.convert_to_tensor(x_ne,dtype=tf.float64), [-1, d]) / 255.
    x_val_po, x_val_ne = tf.reshape(tf.convert_to_tensor(x_val_po,dtype=tf.float64),[-1, d]) / 255., tf.reshape(tf.convert_to_tensor(x_val_ne,dtype=tf.float64), [-1, d]) / 255.

    po_num = x_po.shape[0]
    ne_num = x_ne.shape[0]
    po_vnum = x_val_po.shape[0]
    ne_vnum = x_val_ne.shape[0]
    inte = ne_num // po_num
    intev = ne_vnum // po_vnum

    x_diff = np.zeros([po_num * inte,2, d])
    y_diff = np.zeros([po_num * inte])
    xv_diff = np.zeros([po_vnum * intev,2, d])
    yv_diff = np.zeros([po_vnum * intev])

    index_po,index_ne = tf.random.shuffle(range(po_num)),tf.random.shuffle(range(ne_num))
    indexv_po, indexv_ne = tf.random.shuffle(range(po_vnum)), tf.random.shuffle(range(ne_vnum))
    x_po,y_po = tf.gather(x_po,index_po),tf.gather(y_po,index_po)
    x_ne,y_ne = tf.gather(x_ne,index_ne),tf.gather(y_ne,index_ne)
    x_val_po, y_val_po = tf.gather(x_val_po, indexv_po), tf.gather(y_val_po, indexv_po)
    x_val_ne, y_val_ne = tf.gather(x_val_ne, indexv_ne), tf.gather(y_val_ne, indexv_ne)
    x_po, x_ne = tf.expand_dims(x_po, axis=1), tf.expand_dims(x_ne, axis=1)
    x_val_po, x_val_ne = tf.expand_dims(x_val_po, axis=1), tf.expand_dims(x_val_ne, axis=1)

    for i in range(inte):
        if i < intev / 2:
            x_diff[int(i * po_num):int((i + 1) * po_num), :] = tf.concat([x_po,x_ne[int(i * po_num):int((i + 1) * po_num)]],axis=1)
            y_diff[int(i * po_num):int((i + 1) * po_num)] = 0
        else:
            x_diff[int(i * po_num):int((i + 1) * po_num), :] = tf.concat([x_ne[int(i * po_num):int((i + 1) * po_num)],x_po],axis=1)
            y_diff[int(i * po_num):int((i + 1) * po_num)] = y_po - y_ne[int(i * po_num):int((i + 1) * po_num)]

    for i in range(intev):
        if i < intev / 2:
            xv_diff[int(i * po_vnum):int((i + 1) * po_vnum), :] = tf.concat([x_val_po,x_val_ne[int(i * po_vnum):int((i + 1) * po_vnum)]],axis = 1)
            yv_diff[int(i * po_vnum):int((i + 1) * po_vnum)] = 0
        else:
            xv_diff[int(i * po_vnum):int((i + 1) * po_vnum), :] = tf.concat([x_val_ne[int(i * po_vnum):int((i + 1) * po_vnum)],x_val_po],axis=1)
            yv_diff[int(i * po_vnum):int((i + 1) * po_vnum)] = y_val_po - y_val_ne[
                                                                          int(i * po_vnum):int((i + 1) * po_vnum)]

    x_diff, xv_diff = tf.convert_to_tensor(x_diff, dtype=tf.float64), tf.convert_to_tensor(xv_diff, dtype=tf.float64)
    y_diff, yv_diff = tf.convert_to_tensor(y_diff, dtype=tf.int32), tf.convert_to_tensor(yv_diff, dtype=tf.int32)
    index, index_val = tf.random.shuffle(range(po_num * inte)), tf.random.shuffle(range(po_vnum * intev))
    x_diff, y_diff = tf.gather(x_diff, index), tf.gather(y_diff, index)
    xv_diff, yv_diff = tf.gather(xv_diff, index_val), tf.gather(yv_diff, index_val)
    y_diff = tf.one_hot(y_diff, depth=2)
    yv_diff = tf.one_hot(yv_diff, depth=2)


    model = Network(dimension=d,lamd=lam)

    # earlystop_callback = EarlyStopping(monitor='val_auc', min_delta=0.001, patience=30)
    save = ModelCheckpoint('Weight2/cifar/cifar2_regular'+str(lamd_index) , monitor='val_auc', verbose=1, save_weights_only=True,
                           save_best_only=True)
    earlystop_callback = EarlyStopping(monitor='val_auc', min_delta=0.001, patience=5)
    model.compile(optimizer=optimizer,loss= ,metrics=[tf.keras.metrics.AUC(curve='ROC',name='auc')])
    history = model.fit(x=x_diff,y=y_diff,batch_size=64,epochs= 500 ,validation_data=(xv_diff,yv_diff),callbacks=[earlystop_callback,save])

    all_auc = history.history['auc']
    train_auc = np.mean(all_auc)
    auc.append(train_auc)
    # np.save('Linear_auc/w1a/w1al2_0.'+str(i)+'.npy', train_auc)
print(auc)
np.save('NN_auc/cifar2_regular_all.npy',auc)


