import numpy as np


def Cholesky(A):
	# Cholesky: assume positive definite matrix A
	# A = R^T @ R
	n = A.shape[0]
	R = A.copy()
	R = np.triu(R)
	for k in range(n): # 0 - m-1
		R[k,k:n] /= np.sqrt(R[k,k])
		for j in range(k+1,n):
			R[j,j:] -= R[k,j] * R[k,j:n]
	return R



