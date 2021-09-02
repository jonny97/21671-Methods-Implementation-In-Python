import numpy as np
np.set_printoptions(suppress=True)


# Phase 1 of eigenvalue decomposition
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
			'''
				if H is currently [a w.T;			    w 				K]
				the new matrix is [a (w-2v(v.T @ w)).T; w-2v(v.T @ w)	K - 2vv'K - 2Kvv' + 4vv'Kvv']
				with the following optimal sequence of calculation, total flop count is 4/3n^3
			'''
			H[i+1:,i] -= 2*v*(v.T @ H[i+1:,i])
			#
			# To calculate K - 2vv'K - 2Kvv' + 4vv'Kvv', where K is N by N
			#	T1 = v'K 				  (flop: 2 N^2)
			#   T2 = T1 v = v'Kv 		  (flop: O(N))
			#	T3 = (2v) @ (-T1 + T2*v') (flop: N^2)
			#   
			#	then new K is just K - T3 - T3.T, by symmetry, the cost is N^2 (2 for each triangular entry)
			#	so the total cost is 4N^2, for N from n-1 to 2, which is 4/3 n^3
			K  = H[i+1:,i+1:]
			# T1 = v.T @ K. However, since K is only stored in half, special treatment is as below
			T1 = np.zeros(v.shape)
			for temp_i in range(len(v)):
				for temp_j in range(len(v)):
					T1[temp_i] += v[temp_j] * K[max(temp_i,temp_j),min(temp_i,temp_j)]
			T2 = T1 @ v			
			T3 = np.outer(2*v , -T1 + T2*v.T)

			for temp_i in range(n-i-1):
				for temp_j in range(temp_i+1):
					K[temp_i,temp_j] += (T3[temp_i,temp_j]+T3[temp_j,temp_i])
		else:
			H[i+1:,i:] -= np.outer(2*v , (v.T @ H[i+1:,i:]))
			H[:,i+1:] -=  np.outer((H[:,i+1:] @ v),2*v.T) 

	if Hermitian:
		H = H+np.tril(H,-1).T # transform back to hermitian matrix

	if calc_q:
		Q = np.eye(n)
		for k in range(len(V_list)-1,-1,-1):
			Q[k+1:,:]-= 2*np.outer(V_list[k],(V_list[k] @ Q[k+1:,:]))
		return Q,H
	else:
		return H


if False:
	A = np.random.random((4,4))
	A = A+A.T
	Q,H=Hessenberg(A,Hermitian=0)
	print("nonHermitian Error:",np.max(Q@H@Q.T-A))
	Q,H=Hessenberg(A,Hermitian=1)
	print("Hermitian Error:",np.max(Q@H@Q.T-A))

# flop count: each (v@v.T) operation will take 4 flops.
#  general 10/3 n^3, Hermitian 8/3 n^3, Hermitian utilizing all symmetry 4/3 n^3, 


