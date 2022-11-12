import tensorflow as tf
import numpy as np

# Here is an example of calculating perturbation strength
Auc2_ = []
for i in range(11):
    auc = np.load('Linear_auc/w1a/w1a_0.'+str(i)+'.npy')  # natural AUC
    Auc2_.append(auc)
accmean = np.mean(acc2)
print(accmean)
att_auc = np.load('Attack_auc/w1a/w1a_aveauc.npy')  # adversarial AUC

error = Auc2_-att_auc # [natural_generalization_error,adversarial_generalization_error]
generalization_error = error[0]
generalization_diff = error-generalization_error  # i.e., perturbation strength
print(generalization_diff)




# Here is natural training , test AUC  and attack AUC (Run 10 times)

attack_acc = np.load('Attack_auc/w1a/w1al2_aveauc.npy')
Auc_ = []
for i in range(11):
    auc = np.load('NN_auc/w1a_'+str(i)+'.npy')
    Auc_.append(auc)

Auc_all = np.load('Attack_auc/w1a_ave.npy')
aucmean = np.mean(Auc_)
Final_auc = accmean - Auc_all

np.save('Attack_auc/w1a/w1a.npy', Final_auc)



