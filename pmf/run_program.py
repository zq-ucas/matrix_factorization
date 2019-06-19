import numpy as np 
import csv
import matplotlib.pyplot as plt
from matrix_factorization import pmf

def create_data_matrix(file_loc):
	'''creates a data matrix from file_loc(.txt file including the location)'''

	n, m = return_doc_stats(file_loc)
	print(n, m)
	data_mat = np.full((n, m), np.infty)
	i = 0
	with open(file_loc) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter = ",")
		for row in csv_reader:
			i = int(row[0])
			j = int(row[1])
			data_mat[i - 1][j - 1] = float(row[2]) 
	return data_mat

def return_doc_stats(doc_coll):
	'''returns the number of documements and vocabulary size of the 
	   collection'''

	n_u = 0
	n_v = 0
	vocab_size = 0
	with open(doc_coll) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter = ",")
		for row in csv_reader:
			if int(row[0]) > n_u:
				n_u = int(row[0])
			if int(row[1]) > n_v:
				n_v = int(row[1])

	return n_u, n_v


if __name__ == '__main__':

	X = create_data_matrix("data/ratings.csv")
	sigma_2 = 0.25
	d = 10
	lamb = 1

	model = pmf(X, d, lamb, sigma_2, 250)



