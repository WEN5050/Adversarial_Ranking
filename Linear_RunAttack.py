import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, datasets, optimizers, metrics, layers, callbacks,losses
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from Linear_Train import Linear
from Linear_PGD import LinfPGDAttack






epsilon_list = [0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]  # perturbation bound of experiment related to dimension
# epsilon_list = [0.0,0.010,0.1]  # perturbation bound of experiment related to the magnitude of the weights
addtest_auc = np.zeros([11,3],dtype=np.float64)
add_aveauc = []
add_varauc = []


d = 300  # input_dimension
pertur = np.zeros([len(epsilon_list),2])
for eps_idx in range(len(epsilon_list)):
    auc = []

    po_per = []
    ne_per = []
    for t in range(3):
        per_po = []
        per_ne = []

        x_tes_po, y_test_po = np.load('positivedata/w1a_testx.npy'), np.load('positivedata/w1a_testy.npy')
        x_tes_ne, y_test_ne = np.load('negativedata/w1a_testx.npy'), np.load('negativedata/w1a_testy.npy')

        po_num = x_tes_po.shape[0]
        ne_num = x_tes_ne.shape[0]
        index_po,index_ne = tf.random.shuffle(range(po_num)),tf.random.shuffle(range(ne_num))
        x_tes_po , y_test_po = tf.gather(x_tes_po,index_po) , tf.gather(y_test_po,index_po)
        x_tes_ne,y_test_ne = tf.gather(x_tes_ne,index_ne) , tf.gather(y_test_ne,index_ne)
        print(index_po)
        print(index_ne)
        inte = ne_num // po_num
        eps = epsilon_list[eps_idx]
        print('eps is %.2f' %eps)

        y_adv_diff = np.zeros([po_num * inte])      # rebuild pairwise label
        x_adv_diff = np.zeros([po_num * inte, d])  # rebuild pairwise input

        linear = Linear(d)
        linear.load_weights('Weight2/w1a/w1a')
        linear.compile(optimizer=optimizers.Adam(1e-3),loss=tf.keras.losses.hinge,metrics=[tf.keras.metrics.AUC(curve='ROC', name='auc')])

        Attack = LinfPGDAttack(model=linear, epsilon=eps, alpha = float(eps / 10), k= 10 , random_start=True, batch_size=64)
        for i in range(inte):
            if i == 0:
                y_adv_diff[int(i * po_num):int((i + 1) * po_num)] = y_test_po - y_test_ne[int(i * po_num):int((i + 1) * po_num)]
                x_adpo = np.zeros([po_num,d],dtype=np.float64)
                x_adne = np.zeros([po_num,d],dtype=np.float64)
                x_adpo = Attack.perturb_po(x_tes_po,x_tes_ne[int(i * po_num):int((i + 1) * po_num)],y_adv_diff[int(i * po_num):int((i + 1) * po_num)])  #  perturb positive data
                x_adne = Attack.perturb_ne(x_tes_po,x_tes_ne[int(i * po_num):int((i + 1) * po_num)],y_adv_diff[int(i * po_num):int((i + 1) * po_num)])  #  perturb negative data
                x_adv_diff[int(i * po_num):int((i + 1) * po_num), :] = x_adpo - x_adne


                diff_po = tf.convert_to_tensor(x_adpo - x_tes_po, dtype=tf.float64)
                per_po.append(tf.reduce_min(tf.norm(diff_po,ord=2,axis=1)/d))   # calculate average pixel-level perturbtation of positive data

                diff_ne = tf.convert_to_tensor(x_adne - x_tes_ne[int(i * po_num):int((i + 1) * po_num)],dtype=tf.float64) # calculate average pixel-level perturbtation of negative data
                per_ne.append(tf.reduce_min(tf.norm(diff_ne,ord=2,axis=1)/d))



            else:
                y_adv_diff[int(i * po_num):int((i + 1) * po_num)] = 0
                x_adne = Attack.perturb_po(x_tes_ne[int(i * po_num):int((i + 1) * po_num)],x_tes_po ,y_adv_diff[int(i * po_num):int((i + 1) * po_num)])
                x_adpo = Attack.perturb_ne(x_tes_ne[int(i * po_num):int((i + 1) * po_num)],x_tes_po,y_adv_diff[int(i * po_num):int((i + 1) * po_num)])
                x_adv_diff[int(i * po_num):int((i + 1) * po_num), :] = x_adne - x_adpo
                diff_po = tf.convert_to_tensor(x_adpo - x_tes_po, dtype=tf.float64)
                per_po.append(tf.reduce_min(tf.norm(diff_po, ord=2, axis=1)/d))
                diff_ne = tf.convert_to_tensor(x_adne - x_tes_ne[int(i * po_num):int((i + 1) * po_num)],dtype=tf.float64)
                per_ne.append(tf.reduce_min(tf.norm(diff_ne, ord=2, axis=1)/d))


        print('-'*20)
        po = tf.reduce_mean(per_po)
        print(po)
        ne = tf.reduce_mean(per_ne)
        po_per.append(po)
        ne_per.append(ne)
        print('generate adversarial examples finished.')
        x_adv_diff, y_adv_diff = tf.convert_to_tensor(x_adv_diff, dtype=tf.float64), tf.convert_to_tensor(y_adv_diff, dtype=tf.int32)  # build adversarial pairwise dataset
        index = tf.random.shuffle(range(po_num * inte))


        test_auc = linear.evaluate(x_adv_diff, y_adv_diff)[1]
        auc.append(test_auc)
        print('attack finished.')
        # print(po_per)

    pertur[eps_idx,0] = tf.reduce_mean(po_per)
    pertur[eps_idx,1] = tf.reduce_mean(ne_per)
    addtest_auc[eps_idx,:] = auc
add_aveauc, add_stdauc = np.mean(addtest_auc,axis=1), np.std(addtest_auc,axis=1)
print(add_aveauc)
print(add_stdauc)
np.save('Attack_auc2/aucw1a_allauc.npy',addtest_auc)
np.save('Attack_auc2/aucw1a_aveauc.npy',add_aveauc)
np.save('Attack_auc2/aucw1a_stdauc.npy',add_stdauc)
np.save('Attack_auc2/aucw1a_perl22.npy',pertur)
print(pertur)

