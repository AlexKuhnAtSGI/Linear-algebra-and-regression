import numpy as np
from sklearn.metrics import mean_absolute_error
import pandas as pd

PATH = 'ml-100k/'

fields = ["uid", "mid", "rating", "irrelevant"]
df = pd.read_csv(PATH + 'u.data', header=None, names=fields, sep='\t')
#print(df)

ratings = np.zeros((943, 1682))

count = 0
for row in df.itertuples(index=False):
	ratings[row[0] - 1][row[1] - 1] = row[2]
	
def make_2D(u, s, v, Alice):
	# plurals = np.array([
	# [3,1,2,3],
	# [4,3,4,3],
	# [3,2,1,5],
	# [1,6,5,2],
	# ])
	
	# u, s, v = np.linalg.svd(plurals, full_matrices=True)
	# u=u[:, :2]
	# s=np.diag(s[:2])
	# v=v[:, :2]
	
	# s2 = np.zeros((4,4))
	# s2[0][0] = s[0]
	# s2[1][1] = s[1]
	# s2[2][2] = s[2]
	# s2[3][3] = s[3]
	# s = np.array([s])
	
	Alice2D = np.dot(Alice, u)
	Alice2D = np.dot(Alice2D, np.linalg.inv(s))
	return(Alice2D)
	
def reduce(r,k):
	u, s, v = np.linalg.svd(ratings, full_matrices=True)
	u=u[:, :k]
	s=np.diag(s[:k])
	v=v[:, :k]
	return u,s,v

k = 14
u,s,v = reduce(ratings, k)

#Insufficient time to make prediction function