import numpy as np
import time

class KNN:
    def __euclidian_distance(self, elt1, elt2):
        return np.linalg.norm(elt1 - elt2)

    def learn(self, inputs, expected_outputs):
        self.__train_inputs = inputs
        self.__train_outputs = expected_outputs
        self.__train_number, self.__rows, self.__cols = inputs.shape

    def predict(self, x, k=1):
        vals = []
        for i in range(self.__train_number):
            dict_crt = self.__euclidian_distance(x ,self.__train_inputs[i])
            vals.append(dict_crt)
        argsorted = np.argsort(vals)[:k]
        preds = []
        for idx in argsorted:
            preds.append(self.__train_outputs[ idx ])
      
        return max(set(preds), key=preds.count), argsorted[0]

mnist_images_train = np.load('mnist_images_train.npy')
mnist_labels_train = np.load('mnist_labels_train.npy')

mnist_images_train = mnist_images_train.astype('int16')

X_train = mnist_images_train[:48000]
Y_train = mnist_labels_train[:48000]
X_test = mnist_images_train[48000:]
Y_test = mnist_labels_train[48000:]


print("train shape", X_train.shape)
print("test shape", X_test.shape)

knn = KNN()
knn.learn(X_train, Y_train)

preds = []
times = []
acc   = []
nb_preds = 100
k = 1

for i in range(nb_preds):
    t1 = time.perf_counter() 
    pred, idx = knn.predict(X_test[i], k)
    times.append(time.perf_counter() - t1)

    if pred == Y_test[i]:
        acc.append(1)
    else:
        acc.append(0)
    preds.append(pred)
print("------------------")
print("For k : ", k)
print("Accuracy : ", sum(acc)/len(acc), "%")
print("Total Time : ", sum(times), "s")
print("Average Time : ", sum(times)/len(times), "s")
