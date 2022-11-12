import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, datasets, optimizers, metrics, layers, callbacks
import cv2

#  Select Number 8 and 9


Num1 = 8
Num2 = 9

def Mnist_num(num1,num2):  # Select Number
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    train_filter = np.where((y_train == num1) | (y_train == num2))
    test_filter = np.where((y_test == num1) | (y_test == num2))
    x_train,y_train = x_train[train_filter],y_train[train_filter]
    x_test,y_test = x_test[test_filter],y_test[test_filter]
    return x_train,y_train,x_test,y_test

x_train,y_train,x_test,y_test = Mnist_num(Num1,Num2)
y_train,y_test = y_train.astype('int32'), y_test.astype('int32')
train_filter1,test_filter1 = np.where(y_train == Num1),np.where(y_test == Num1)
train_filter2,test_filter2 = np.where(y_train == Num2),np.where(y_test == Num2)
y_train[train_filter1],y_test[test_filter1] = 1, 1  # number 8 denotes positive data
y_train[train_filter2],y_test[test_filter2] = 0, 0  # number 9 denotes positive data

# x_8,y_8 = tf.convert_to_tensor(x_train[train_filter1]),tf.convert_to_tensor(y_train[train_filter1])
# x_9,y_9 = tf.convert_to_tensor(x_train[train_filter2]) ,tf.convert_to_tensor(y_train[train_filter2])
# print(x_8.shape,y_8.shape)
# print(x_9.shape,y_9.shape)
# print(y_8)
# print(y_9)

# x_num1,y_num1 = tf.convert_to_tensor(x_train[train_filter1]),tf.convert_to_tensor(y_train[train_filter1])
# x_num2,y_num2 = tf.convert_to_tensor(x_train[train_filter2]),tf.convert_to_tensor(y_train[train_filter2])
# #
# x_num1_tr,y_num1_tr = x_num1[:1000] ,y_num1[:1000]
# x_num2_tr,y_num2_tr = x_num2[:6000] ,y_num2[:6000]
#
# #
# x_num1_val,y_num1_val = x_num1[1000:1123] ,y_num1[1000:1123]
# x_num2_val,y_num2_val = x_num2[6000:6742] , y_num2[6000:6742]
# print(x_num1_val.shape,y_num1_val.shape)
# print(x_num2_val.shape,y_num2_val.shape)
# print(y_num1_val,y_num2_val)


x_num1_te,y_num1_te = tf.convert_to_tensor(x_test[test_filter1][:189]) , tf.convert_to_tensor(y_test[test_filter1][:189])
x_num2_te,y_num2_te = tf.convert_to_tensor(x_test[test_filter2]) , tf.convert_to_tensor(y_test[test_filter2])
print(x_num1_te.shape,y_num1_te.shape)
print(x_num2_te.shape,y_num2_te.shape)
print(y_num1_te,y_num2_te)
# a.tolist   a.count(0)


#
np.save('./positivedata/fashion_testx.npy',x_num1_te)
np.save('./positivedata/fashion_testy.npy',y_num1_te)
#

np.save('./negativedata/fashion_testx.npy',x_num2_te)
np.save('./negativedata/fashion_testy.npy',y_num2_te)
#
# np.save('./positivedata/fashion_valx.npy',x_num1_val)
# np.save('./positivedata/fashion_valy.npy',y_num1_val)
# #
#
# np.save('./negativedata/fashion_valx.npy',x_num2_val)
# np.save('./negativedata/fashion_valy.npy',y_num2_val)