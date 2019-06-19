import numpy as np  

class NMF:
	def __init__(self, X, K, T):
		'''X: data matrix, K: rank of matrix decompositions, 
		   T: number of iterations'''

		self.X = X
		self.K = K
		self.T = T

		self.N = X.shape[0] #number of words
		self.M = X.shape[1] #number of documents

		self.obj = []
		self.W, self.H = self.run_updates()


	def run_updates(self):
		'''updates the matrix W and H multiplicatively'''

		W_0 = np.random.uniform(1, 2, (self.N, self.K))
		H_0 = np.random.uniform(1, 2, (self.K, self.M))

		for t in range(self.T):
			print("iteration number:", t)
			H_1 = self.update_H(W_0, H_0)
			W_1 = self.update_W(W_0, H_1)
			objective = self.compute_objective(W_1, H_1)
			self.obj.append(objective)
			H_0 = H_1
			W_0 = W_1

		return W_0, H_0

	def update_H(self, W, H):
		'''updates the matrix H'''

		WH = np.dot(W, H) + 1e-16
		X_WH = self.X / WH
		W_TX = np.dot(W.T, X_WH)
		denom = np.sum(W, axis = 0, keepdims = True)
		return H * W_TX / denom.T

	def update_W(self, W, H):
		'''updates the matrix W'''

		WH = np.dot(W, H) + 1e-16
		X_WH = self.X / WH
		HX = np.dot(X_WH, H.T)
		denom = np.sum(H, axis = 1, keepdims = True)
		return W * HX / denom.T


	def compute_objective(self, W, H):
		'''computes the value of the objective function'''

		WH = np.dot(W, H) + 1e-16
		obj_mat = self.X * np.log(1 / WH) + WH - 1e-16 

		return np.sum(obj_mat)


	def get_objective(self):
		'''return the value of the objective function'''
		return self.obj



