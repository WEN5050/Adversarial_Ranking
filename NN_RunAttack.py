import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, datasets, optimizers, metrics, layers, callbacks,losses
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from NN_Train_Lambda import Network
from NN_PGD import LinfPGDAttack







# epsilon_list = [0.0,0.010,0.020,0.030,0.040,0.050,0.060,0.070,0.080,0.090,0.1] # perturbation bound of experiment related to dimension
epsilon_list = [0.0,0.08,0.15] # perturbation bound of experiment related to the magnitude of the weights
addtest_auc = np.zeros([11,10],dtype=np.float64)
addave_auc = np.zeros([3,11], dtype=np.float64)
addvar_auc = np.zeros([3,11], dtype=np.float64)
lamd = [0.00,0.002,0.004,0.006,0.008,0.01,0.012,0.014,0.016,0.018,0.02]


d = 3072 # dimension

for n in range(len(epsilon_list)):
    eps = epsilon_list[n]

    var_auc = []
    ave_auc = []
    for lamd_index  in range(len(lamd)):
        aucc = []
        lam = lamd[lamd_index]
        for t in range(10):
            per_po = []
            per_ne = []

            x_tes_po, y_test_po = np.load('positivedata/cifar_testx.npy'), np.load('positivedata/cifar_testy.npy')
            x_tes_ne, y_test_ne = np.load('negativedata/cifar_testx.npy'), np.load('negativedata/cifar_testy.npy')
            x_tes_po,x_tes_ne = tf.reshape(tf.convert_to_tensor(x_tes_po,dtype=tf.float64),[-1,d])/255.,tf.reshape(tf.convert_to_tensor(x_tes_ne,dtype=tf.float64),[-1,d])/255.
            # x_tes_po,x_tes_ne = normlization(x_tes_po),normlization(x_tes_ne)
            po_num = x_tes_po.shape[0]
            ne_num = x_tes_ne.shape[0]
            index_po,index_ne = tf.random.shuffle(range(po_num)),tf.random.shuffle(range(ne_num))
            x_tes_po , y_test_po = tf.gather(x_tes_po,index_po) , tf.gather(y_test_po,index_po)
            x_tes_ne,y_test_ne = tf.gather(x_tes_ne,index_ne) , tf.gather(y_test_ne,index_ne)

            inte = ne_num // po_num

            print('eps is %.2f' %eps)

            y_adv_diff = np.zeros([po_num * inte])
            x_adv_diff = np.zeros([po_num * inte,2, d])

            network = Network(d,lamd=lam)
            network.load_weights('Weight2/cifar/cifar2_regular'+str(lamd_index))
            network.compile(optimizer=optimizers.Adam(1e-4),loss=network.myloss,metrics=[tf.keras.metrics.AUC(curve='ROC', name='auc')])

            network2 = Network(d, lamd=0)
            network2.load_weights('Weight2/cifar/cifar2_regular0' )
            network2.compile(optimizer=optimizers.Adam(1e-4), loss=network2.myloss,
                            metrics=[tf.keras.metrics.AUC(curve='ROC', name='auc')])

            Attack = LinfPGDAttack(model=network2, epsilon=eps, alpha = float(eps / 10), k= 10 , random_start=True, batch_size=64)
            for i in range(inte):
                if i < inte / 2:
                    y_adv_diff[int(i * po_num):int((i + 1) * po_num)] = 0
                    x_adpo = np.zeros([po_num,d],dtype=np.float64)
                    x_adne = np.zeros([po_num,d],dtype=np.float64)
                    x_adpo = Attack.perturb_po(x_tes_po,x_tes_ne[int(i * po_num):int((i + 1) * po_num)],y_adv_diff[int(i * po_num):int((i + 1) * po_num)])
                    x_adne = Attack.perturb_ne(x_tes_po,x_tes_ne[int(i * po_num):int((i + 1) * po_num)],y_adv_diff[int(i * po_num):int((i + 1) * po_num)])


                    x_adpo = tf.expand_dims(x_adpo, axis=1)
                    x_adne = tf.expand_dims(x_adne, axis=1)
                    x_adv_diff[int(i * po_num):int((i + 1) * po_num), :] = tf.concat([x_adpo, x_adne], axis=1)

                    # diff_po = tf.convert_to_tensor(x_adpo - x_tes_po, dtype=tf.float64)  # calculate average pixel-level perturbation
                    # per_po.append(tf.reduce_max(tf.norm(diff_po,ord=2,axis=1)/d))
                    #
                    # diff_ne = tf.convert_to_tensor(x_adne - x_tes_ne[int(i * po_num):int((i + 1) * po_num)],dtype=tf.float64)
                    # per_ne.append(tf.reduce_max(tf.norm(diff_ne,ord=2,axis=1)/d))



                else:
                    y_adv_diff[int(i * po_num):int((i + 1) * po_num)] = y_test_po - y_test_ne[int(i * po_num):int((i + 1) * po_num)]
                    x_adne = Attack.perturb_po(x_tes_ne[int(i * po_num):int((i + 1) * po_num)],x_tes_po ,y_adv_diff[int(i * po_num):int((i + 1) * po_num)])
                    x_adpo = Attack.perturb_ne(x_tes_ne[int(i * po_num):int((i + 1) * po_num)],x_tes_po,y_adv_diff[int(i * po_num):int((i + 1) * po_num)])
                    x_adpo = tf.expand_dims(x_adpo, axis=1)
                    x_adne = tf.expand_dims(x_adne, axis=1)
                    x_adv_diff[int(i * po_num):int((i + 1) * po_num), :] = tf.concat([x_adne, x_adpo], axis=1)
                    # diff_po = tf.convert_to_tensor(x_adpo - x_tes_po, dtype=tf.float64)
                    # per_po.append(tf.reduce_max(tf.norm(diff_po, ord=2, axis=1)/d))
                    # diff_ne = tf.convert_to_tensor(x_adne - x_tes_ne[int(i * po_num):int((i + 1) * po_num)],dtype=tf.float64)
                    # per_ne.append(tf.reduce_max(tf.norm(diff_ne, ord=2, axis=1)/d))

                    # per_ne.append(np.max(np.max(np.abs(x_adne - x_tes_ne[int(i * po_num):int((i + 1) * po_num)]),axis=1)))
            # print('-'*20)
            # po = tf.reduce_max(per_po)
            # print(po)
            # ne = tf.reduce_max(per_ne)
            # po_per.append(po)
            # ne_per.append(ne)
            print('generate adversarial examples finished.')
            x_adv_diff, y_adv_diff = tf.convert_to_tensor(x_adv_diff, dtype=tf.float64), tf.convert_to_tensor(y_adv_diff, dtype=tf.int32)
            index = tf.random.shuffle(range(po_num * inte))
            # x_adv_diff, y_adv_diff = tf.gather(x_adv_diff, index[:3000]), tf.gather(y_adv_diff, index[:3000])
            x_adv_diff, y_adv_diff = tf.gather(x_adv_diff, index), tf.gather(y_adv_diff, index)
            y_adv_diff = tf.one_hot(y_adv_diff, depth=2)

            test_auc = network.evaluate(x_adv_diff, y_adv_diff)[1]
            aucc.append(test_auc)
            print('attack finished.')
            # print(po_per)

        # pertur[eps_idx,0] = tf.reduce_mean(po_per)
        # pertur[eps_idx,1] = tf.reduce_mean(ne_per)

        ave_auc.append(np.mean(aucc))
        var_auc.append(np.std(aucc))
    addave_auc[n,:] = ave_auc
    addvar_auc[n,:] = var_auc
# add_aveauc, add_stdauc = np.mean(addtest_auc,axis=1), np.std(addtest_auc,axis=1)

# np.save('Attack_auc2/cifa_allauc.npy',addtest_auc)
print(addave_auc)
np.save('Attack_auc2/cifa_regular_aveauc.npy',addave_auc)
np.save('Attack_auc2/cifa_regular_stdauc.npy',addvar_auc)

# print('attack is finished.....')








# d = 22
# x_tes_po, y_test_po = np.load('positivedata/ijcnn1_testx.npy'), np.load('positivedata/ijcnn1_testy.npy')
# x_tes_ne, y_test_ne = np.load('negativedata/ijcnn1_testx.npy'), np.load('negativedata/ijcnn1_testy.npy')
# po_num = x_tes_po.shape[0]
# ne_num = x_tes_ne.shape[0]
# inte = ne_num // po_num
# x_diff = np.zeros([po_num*inte,d])
# y_diff = np.zeros([po_num*inte])
#
# for i in range(inte):
#     if i < inte/2:
#         x_diff[int(i*po_num):int((i+1)*po_num),:] = x_tes_po - x_tes_ne[int(i*po_num):int((i+1)*po_num)]
#         y_diff[int(i*po_num):int((i+1)*po_num)] = y_test_po - y_test_ne[int(i*po_num):int((i+1)*po_num)]
#     else:
#         x_diff[int(i * po_num):int((i + 1) * po_num), :] = x_tes_ne[int(i * po_num):int((i + 1) * po_num)] - x_tes_po
#         y_diff[int(i * po_num):int((i + 1) * po_num)] = 0
#
# x_diff = tf.convert_to_tensor(x_diff,dtype=tf.float64)
# y_diff = tf.convert_to_tensor(y_diff,dtype=tf.int32)
# index = tf.random.shuffle(range(po_num*inte))
# x_diff,y_diff = tf.gather(x_diff,index), tf.gather(y_diff,index)
#
# network = Network(d)
# network.load_weights('Weight/ijcnn1/ijcnn1_0.0')
# network.compile(optimizer=optimizers.Adam(lr=1e-3), loss=tf.keras.losses.hinge,metrics=[tf.keras.metrics.AUC(curve='ROC', name='auc')])
# test_auc = network.evaluate(x_diff, y_diff)[1]
# print(test_auc)
