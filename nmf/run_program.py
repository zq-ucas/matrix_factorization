import numpy as np 
import csv
from NMF import NMF 
import matplotlib.pyplot as plt

def create_data_matrix(file_loc):
	'''creates a data matrix from file_loc(.txt file including the location)'''

	n, m = return_doc_stats(file_loc)
	print(n, m)
	data_mat = np.zeros((n, m))
	i = 0
	with open(file_loc) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter = ",")
		for row in csv_reader:
			for items in row:
				item_split = items.split(":")
				j = int(item_split[0]) - 1
				value = int(item_split[1])
				data_mat[i][j] = value
			i += 1

	return data_mat.T

def return_doc_stats(doc_coll):
	'''returns the number of documements and vocabulary size of the 
	   collection'''

	n_docs = 0
	vocab_size = 0
	with open(doc_coll) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter = ",")
		for row in csv_reader:
			for idx_ctn in row:
				idx_ctn_li = idx_ctn.split(":")
				if int(idx_ctn_li[0]) > vocab_size:
					vocab_size = int(idx_ctn_li[0])
			n_docs += 1

	return n_docs, vocab_size


if __name__ == '__main__':

	X = create_data_matrix("data/nyt_data.txt")
	model = NMF(X, 25, 100)
	results = model.get_objective()
	fig = plt.figure()
	axes = fig.add_subplot(1,1,1)
	axes.plot(np.linspace(1, len(results), len(results)), results)
	axes.set_xlabel("iteration number")
	axes.set_ylabel("Objective function value")
	axes.set_title("Objective function per iteration")
	plt.show()


