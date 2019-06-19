import numpy as np
import copy
import numpy.linalg as lin 
import matplotlib.pyplot as plt 


class pmf:

	def __init__(self, X, k, lambd, sigma, T):
		'''X: data matrix 
		   k: rank of the matrix decompositions
		   lambda: parameter for the mulitivariate-gaussion prior on u and v
		   sigma: variance of X_{ij}
		   T: number of iterations for co-ordinate ascent'''

		self.X = X
		self.n = X.shape[0]
		self.m = X.shape[1]
		self.k = k
		self.lambd = lambd
		self.sigma = sigma
		self.T = T
		#print(self.X)
		#print(self.m)
		self.U, self.V = self.compute_UV()
		self.log_likelihoods = []


	def compute_UV(self):
		'''computes the decompositions U and V'''

		L_values = [] #list containing the og likelihood values

		mean_u = np.zeros(self.k)
		cov_u = 1 / self.sigma * np.identity(self.k)
		U_0 = np.random.multivariate_normal(mean_u, cov_u, self.n)

		mean_v = np.zeros(self.k)
		cov_v = 1 / self.sigma * np.identity(self.k)
		V_0 = np.random.multivariate_normal(mean_v, cov_v, self.m).T

		for t in range(self.T):

			U_1 = self.update_U(V_0)
			V_1 = self.update_V(U_1)
			L = self.evaluate_joint_likelihood(U_1, V_1)
			#print(L)
			self.log_likelihoods.append(L)
			if t > 0:
				diff = L_values[t] - L_values[t - 1]
				if diff < 0:
					print("Error")
				else:
					print(diff)
			U_0 = U_1
			V_0 = V_1

		fig = plt.figure()
		axes = fig.add_subplot(1,1,1)
		axes.set_xlabel("# iteration")
		axes.set_ylabel("$\mathcal{L}$")
		axes.plot(np.linspace(1, self.T, self.T), L_values)
		axes.set_title("Convergence of the algorithm")
		plt.show()

		return U_0, V_0

	def update_U(self, V):
		'''updates the user matrix U'''

		U = np.zeros((self.n, self.k))
		for i in range(self.n):
			x_i = self.X[[i], :]
			V_copy = copy.deepcopy(V.T)
			present_i = np.where(x_i != np.inf, 1, 0).T

			V_i = V_copy * present_i
			V_VT = np.dot(V_i.T, V_i)
			x_i_mod = np.where(x_i != np.inf, x_i, 0).T
			XV = x_i_mod * V_i
			c1 = lin.inv(self.sigma * self.lambd * np.identity(self.k) + V_VT)
			c2 = np.sum(XV.T, axis = 1, keepdims = True)
			u_i = np.matmul(c1, c2).T
			U[i] = u_i

		return U

	def update_V(self, U):
		'''updates the movie matrix V'''

		V = np.zeros((self.k, self.m)).T
		for j in range(self.m):
			x_j = self.X[:,[j]]
			U_copy = copy.deepcopy(U)
			present_j = np.where(x_j != np.inf, 1, 0)
			U_j = U_copy * present_j 
			U_UT = np.dot(U_j.T, U_j)
			x_j_mod = np.where(x_j != np.inf, x_j, 0)
			XU = x_j_mod * U_j
			c1 = lin.inv(self.sigma * self.lambd * np.identity(self.k) + U_UT)
			c2 = np.sum(XU.T, axis = 1, keepdims = True)
			v_j = np.matmul(c1, c2).T
			V[j] = v_j

		return V.T
			
	def evaluate_joint_likelihood(self, U, V):
		'''evaluated the log likelihood function'''

		M = self.X - np.matmul(U, V)
		M = np.where(M != np.infty, M, 0)
		M_norm = lin.norm(M) ** 2
		u_norm = lin.norm(U) ** 2
		v_norm = lin.norm(V) ** 2 

		return - 1 / self.sigma * M_norm - self.lambd * u_norm / 2 -\
		self.lambd /2 * v_norm

	def get_U(self):

		return self.U

	def get_V(self):

		return self.V

	def get_log_likelihood(self):

		return self.log_likelihoods

		
















