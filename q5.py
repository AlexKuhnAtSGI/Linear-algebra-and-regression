import numpy as np
from mnist import MNIST
import random
import time
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

#As reference: each digit is encoded as a 28*28 grayscale image, and then we flatten that to a vector with 784 elements

#We were not told to include the MNIST data, so I have not. I had all 4 files in a 'samples' directory within my a1 folder:
#I access it on line 112 mndata = MNIST('samples'). If you want to run this, you'll need to make sure that line is working.



def basis_getter(X_train, y_train, BASIS_SIZE):
	A = {
		0: np.zeros((784, 0)),
		1: np.zeros((784, 0)),
		2: np.zeros((784, 0)),
		3: np.zeros((784, 0)),
		4: np.zeros((784, 0)),
		5: np.zeros((784, 0)),
		6: np.zeros((784, 0)),
		7: np.zeros((784, 0)),
		8: np.zeros((784, 0)),
		9: np.zeros((784, 0))
	}

	
	used_indices = []
	
	
	for i in range (10):
		count = 0
		lst = []
		while count < 600:
			index = random.randrange(0, len(y_train))
			if ((y_train[index] == i) and (index not in used_indices)):
				lst.append(X_train[index])
				used_indices.append(index)
				count += 1
		A[i] = lst
	
	# for i in range (10):
		# for j in range (BASIS_SIZE):
			# print(mndata.display(A[i][j]))
			
	for i in range (10):
		#turning A into an array of 784*BASIS_SIZE column vectors
		A[i] = np.linalg.svd(np.asarray(A[i]).transpose())[0][:, :BASIS_SIZE]
		#print(A[i].shape)
	

	
	return A
	
	
def classify(X_test, BASIS_SIZE = 10):
	X_train, y_train = mndata.load_training()

	start = time.time()
	U = basis_getter(X_train, y_train, BASIS_SIZE)
	end = time.time()
	#print("Took this long to train:", end - start)
	
	I = np.identity(784)
	prd_class = 0
	pred = []
	
	progress = 0
	
	for X in X_test:
		residual = 2147483647
		#print(X.transpose())
		X = X.T
		#print(X)
		#print(X_np.shape)
		
		for i in range(10):
			res_term = I - np.dot(U[i], U[i].T)
			res_term = np.dot(res_term, X)
			curr_residual = np.linalg.norm(x=res_term, ord=2)
			
			#print(curr_residual)
			
			if curr_residual < residual:
				residual = curr_residual
				prd_class = i
				
		pred.append(prd_class)
		# progress += 1
		# if progress % 5 == 0:
			# print (progress, "predictions so far...")
		
	return np.asarray(pred)
	
def score(y_pred, y_actual):
	score = y_pred == y_actual
	return np.sum(score) / len(y_pred)
	
def graph(accuracies, bases):
	fig=plt.figure()
	# ax=fig.add_axes([0,0,1,1])
	# ax.scatter(bases, accuracies, color='b', marker ='x')
	# ax.plot(bases, accuracies, color='r')
	# ax.set_xlabel('# of Basis Images')
	# ax.set_ylabel('Classification Percentage')
	
	plt.scatter(bases, accuracies, color='b', marker ='x')
	plt.plot(bases, accuracies, color='r')
	plt.xlabel('# of Basis Images')
	plt.ylabel('Classification Percentage')
	
	plt.show()
	
	
	
mndata = MNIST('samples')
X_test, y_test = mndata.load_testing()
X_test = np.asarray(X_test)

accuracies = []
bases = []

for b in range(2, 51, 2):
	X_test, y_test = shuffle(X_test, y_test)
	y_pred = classify(X_test[0:500], BASIS_SIZE=b)
	acc = score(y_pred, y_test[0:500])
	accuracies.append(acc)
	bases.append(b)

graph(accuracies, bases)