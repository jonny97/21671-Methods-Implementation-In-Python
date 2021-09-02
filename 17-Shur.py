import numpy as np
np.set_printoptions(suppress=True)

def Hessenberg(A,calc_q = True,Hermitian = False):
	# Hessenberg: A square, A = Q.* @ T @ Q
	# Via unitary transformations
	n = A.shape[0]
	Q = np.eye(n)
	if Hermitian:
		H = np.copy(np.tril(A)) # if Hermitian, only lower triangular is needed
	else:
		H = np.copy(A)
	
	V_list = []
	for i in range(n-2):
		v = H[i+1:,i].copy()
		v[0] = v[0] + (2*(v[0]>0)-1) * np.linalg.norm(v,2)
		v = v / np.linalg.norm(v,2)
		V_list.append(v)

		# Apply Q on the right: H v v^T 
		# note that the full column will change!
		if Hermitian:
			H[i+1:,i] -= 2*v*(v.T @ H[i+1:,i])
			H[i:,i+1:] = H[i+1:,i].T
			H[i+1:,i+1:] -= np.outer(2*v , (v.T @ H[i+1:,i+1:]))
			H[i+1:,i+1:] -=  np.outer((H[i+1:,i+1:] @ v),2*v.T) 
		else:
			H[i+1:,i:] -= np.outer(2*v , (v.T @ H[i+1:,i:]))
			H[:,i+1:] -=  np.outer((H[:,i+1:] @ v),2*v.T) 


	if calc_q:
		Q = np.eye(n)
		for k in range(len(V_list)-1,-1,-1):
			Q[k+1:,:]-= 2*np.outer(V_list[k],(V_list[k] @ Q[k+1:,:]))
		return Q,H
	else:
		return H

#test
# A = np.random.random((4,4))
# A = A+A.T
# Q,H=Hessenberg(A,Hermitian=1)
# print(np.max(Q@H@Q.T-A))

# flop count: each (v@v.T) operation will take 4 flops.
#  general 10/3 n^3, Hermitian 8/3 n^3, Hermitian utilizing all symmetry 4/3 n^3, 





# Phase 2


