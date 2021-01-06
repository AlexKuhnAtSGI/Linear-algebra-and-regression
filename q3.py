import numpy as np
from astropy.table import Table

A = np.array([
[1,2,3],
[2,3,4],
[4,5,6],
[1,1,1]])
b = np.array([1,1,1,1]).transpose()

def least_squares():
	x=np.linalg.lstsq(A,b,rcond=None)[0]
	print("Part 1, least squares solution: ", x.flatten(), '\n')

def calc_step(x):
	return np.dot(A.T, np.dot(A, x)) - np.dot(A.T, b)

def gradient():
	delta = 0.01
	epsilons = [0.01,0.05,0.1,0.15,0.2,0.25,0.5]
	xvals = []
	counts = []
	
	for epsy in epsilons:
		count = 0
		x = np.random.rand(A.shape[1])
		step = calc_step(x)
		
		with np.errstate(invalid='raise'):
			while (np.linalg.norm(x=step, ord=2) > delta):
				try:
					x = x - np.dot(epsy, step)
					step = calc_step(x)
					count += 1
				except FloatingPointError:
					#print("value", epsy, "produced error")
					#print(x)
					break
					
		xvals.append(x)
		counts.append(count)
	
	t = Table(names=('x-vals', 'epsilon', 'iterations'))
	t['x-vals'] = t['x-vals'].astype(np.ndarray)
	count = 0
	for val in xvals:
		t.add_row((val, epsilons[count], counts[count]))
		count += 1
	
	print("Part 2, gradient descent using different values of epsilon:\n", t)
least_squares()
gradient()