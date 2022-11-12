import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, datasets, optimizers, metrics, layers, callbacks

# Change Label Y (if y-y' == 1, Y==1, else, Y==0)



# train_filter = np.where((y_train == num1) | (y_train == num2))

y = np.load('Numpy_Data/diabetes_y.npy')


filter1 = np.where(y == 1)
y[filter1] = 0
filter2 = np.where(y == -1)
y[filter2] = 1
np.save('Numpy_Data/diabetes_y.npy',y)
# y_test = np.load('Numpy_Data/cod_rna_ytest.npy')
# y_val = np.load('Numpy_Data/ijcnn1_yval.npy')

# filter1,filter2 = np.where(y == -1),np.where(y_test == -1)
# # filter3 = np.where(y_val == -1)
# # filter1 = np.where(y ==-1)
# y[filter1] = 0
# y_test[filter2] = 0
# # y_val[filter3] = 0
# print(y_val.shape)

# np.save('./Numpy_Data/cod_rna_ytrain.npy',y)
# np.save('./Numpy_Data/cod_rna_ytest.npy',y_test)
# np.save('./Numpy_Data/ijcnn1_yval.npy',y_val)
