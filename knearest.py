import utils as u
import plot_digits as plotd
import run_knn as knn
import numpy as np
import matplotlib.pyplot as plt

train_inputs, train_targets = u.load_train()
#valid_inputs, valid_targets = u.load_valid()
valid_inputs, valid_targets = u.load_test()
#plotd.plot_digits(train_inputs)

valid_results = np.zeros((5,len(valid_targets)))
result = np.zeros((5,1))
valid_results[0,:] = knn.run_knn(1,train_inputs.T,train_targets.T,valid_inputs.T).T
valid_results[1,:] = knn.run_knn(3,train_inputs.T,train_targets.T,valid_inputs.T).T
valid_results[2,:] = knn.run_knn(5,train_inputs.T,train_targets.T,valid_inputs.T).T
valid_results[3,:] = knn.run_knn(7,train_inputs.T,train_targets.T,valid_inputs.T).T
valid_results[4,:] = knn.run_knn(9,train_inputs.T,train_targets.T,valid_inputs.T).T

for y in range(0,5):
    for x in range(0, len(valid_targets)):
        if round(valid_results[y, x]) == round(valid_targets[x, 0]):
            result[y] += 1

classcal = result/len(valid_targets)
print classcal


plt.bar([1,3,5,7,9],classcal,alpha=0.4,color='r')
plt.xlabel('K')
plt.ylabel('Classification rate (# correctly predicted/ total # of data')
plt.title('Classification rate for values of K using the test set')
plt.xticks([1,3,5,7,9], ('1','3','5','7','9'))
plt.show()




