import tensorflow as tf
import numpy as np

from tensorflow.keras import Model, datasets, optimizers, metrics, layers, callbacks,losses
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from  tensorflow.keras.metrics import AUC

# All datasets can be download from LIB-SVM and UCI
# Our bipartite ranking function : f(x,x') = <w,x-x'>
# We build pairwise data : random match positive and negative data,
# then new input == positive_input - negative_input ; new label = positive_y - negative_y
# or new input == negative_input - positive_input; new label = negative_y - positive_y



class Linear(Model):  # Linear models
    def __init__(self,dimension):
        super(Linear,self).__init__()
        self.input_layer = layers.Input((dimension))  # ,kernel_regularizer=tf.keras.regularizers.l1(0.01)  L_1regularizer

        self.out = layers.Dense(1,activation=tf.nn.sigmoid)

    def call(self, inputs, training=None, mask=None):

        out = self.out(inputs)

        return out





auc = []
for i in range(11):
    print(i)
    d = 300  # dimension
    optimizer = optimizers.Adam(lr=1e-3)  # lr
    x_po, y_po = np.load('positivedata/w1a_trainx.npy'), np.load('positivedata/w1a_trainy.npy')
    x_ne, y_ne = np.load('negativedata/w1a_trainx.npy'), np.load('negativedata/w1a_trainy.npy')
    x_val_po, y_val_po = np.load('positivedata/w1a_valx.npy'), np.load('positivedata/w1a_valy.npy')
    x_val_ne, y_val_ne = np.load('negativedata/w1a_valx.npy'), np.load('negativedata/w1a_valy.npy')


    po_num = x_po.shape[0]
    ne_num = x_ne.shape[0]
    po_vnum = x_val_po.shape[0]
    ne_vnum = x_val_ne.shape[0]
    inte = ne_num // po_num
    intev = ne_vnum // po_vnum
    x_diff = np.zeros([po_num * inte, d])  # new pairwise training input
    y_diff = np.zeros([po_num * inte])    # new pairwise training label
    xv_diff = np.zeros([po_vnum * intev, d])  # new pairwise validation input
    yv_diff = np.zeros([po_vnum * intev])   # new pairwise validation label

    index_po,index_ne = tf.random.shuffle(range(po_num)),tf.random.shuffle(range(ne_num))   # shuffle  positive data
    indexv_po, indexv_ne = tf.random.shuffle(range(po_vnum)), tf.random.shuffle(range(ne_vnum))  # shuffle  negative data
    x_po,y_po = tf.gather(x_po,index_po),tf.gather(y_po,index_po)
    x_ne,y_ne = tf.gather(x_ne,index_ne),tf.gather(y_ne,index_ne)
    x_val_po, y_val_po = tf.gather(x_val_po, indexv_po), tf.gather(y_val_po, indexv_po)
    x_val_ne, y_val_ne = tf.gather(x_val_ne, indexv_ne), tf.gather(y_val_ne, indexv_ne)
    for i in range(inte):  # build new training dataset
        if i < inte/2 :   # positive_input and negative_input , new label y == 1
            x_diff[int(i * po_num):int((i + 1) * po_num), :] = x_po - x_ne[int(i * po_num):int((i + 1) * po_num)]
            y_diff[int(i * po_num):int((i + 1) * po_num)] = y_po - y_ne[int(i * po_num):int((i + 1) * po_num)]
        else:  # match negative_input and positive_input , new label y == 0
            x_diff[int(i * po_num):int((i + 1) * po_num), :] = x_ne[int(i * po_num):int((i + 1) * po_num)] - x_po
            y_diff[int(i * po_num):int((i + 1) * po_num)] = 0

    for i in range(intev):  # build new validation dataset
        if i < intev/2:    # positive_input and negative_input , new label y == 1
            xv_diff[int(i * po_vnum):int((i + 1) * po_vnum), :] = x_val_po - x_val_ne[
                                                                             int(i * po_vnum):int((i + 1) * po_vnum)]
            yv_diff[int(i * po_vnum):int((i + 1) * po_vnum)] = y_val_po - y_val_ne[
                                                                          int(i * po_vnum):int((i + 1) * po_vnum)]
        else:  # match negative_input and positive_input , new label y == 0
            xv_diff[int(i * po_vnum):int((i + 1) * po_vnum), :] = x_val_ne[
                                                                  int(i * po_vnum):int((i + 1) * po_vnum)] - x_val_po
            yv_diff[int(i * po_vnum):int((i + 1) * po_vnum)] = 0

    x_diff, xv_diff = tf.convert_to_tensor(x_diff, dtype=tf.float64), tf.convert_to_tensor(xv_diff, dtype=tf.float64)
    y_diff, yv_diff = tf.convert_to_tensor(y_diff, dtype=tf.int32), tf.convert_to_tensor(yv_diff, dtype=tf.int32)
    index, index_val = tf.random.shuffle(range(po_num * inte)), tf.random.shuffle(range(po_vnum * intev))  # shuffle new input-output data
    x_diff, y_diff = tf.gather(x_diff, index), tf.gather(y_diff, index)
    xv_diff, yv_diff = tf.gather(xv_diff, index_val), tf.gather(yv_diff, index_val)

# Training process
    model = Linear(dimension=d)


    save = ModelCheckpoint('Weight2/w1a/w1a' , monitor='val_auc', verbose=1, save_weights_only=True,
                           save_best_only=True)
    earlystop_callback = EarlyStopping(monitor='val_auc', min_delta=0.001, patience=5)
    model.compile(optimizer=optimizer,loss=tf.keras.losses.hinge ,metrics=[tf.keras.metrics.AUC(curve='ROC', name='auc')])
    history = model.fit(x=x_diff,y=y_diff,batch_size=64,epochs= 500 ,validation_data=(xv_diff,yv_diff),callbacks=[earlystop_callback,save])

    all_auc = history.history['auc']
    train_auc = np.mean(all_auc)
    auc.append(train_auc)


np.save('Linear_auc2/aucw1a.npy',auc)


