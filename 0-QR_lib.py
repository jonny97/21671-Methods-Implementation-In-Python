'''
	fix on March 31: 
				backsolve:	x is now a copy of b
							added backsolve for lower triangular matrix
	fix on Feb 27: making reduced Householder QR now work
				   by keeping track of a list of v and form Q at the very end
'''


import numpy as np

def norm_2(v):
	sum = 0.0
	for i in range(v.shape[0]):
		sum+=v[i]**2
	return sum**0.5

def backsolve(R,b,upper=True):
	# Assuming R is upper triangular, solves R x = b, where R,b is given.
	# if upper if False, we assume R is lower triangular
	n = R.shape[1]
	x = b[:n].copy()
	if upper:
		for i in range(n-1,-1,-1): # i ranging from n-1 to 0
			x[i] /= R[i,i]
			for j in range(i):
				x[j] -= x[i] * R[j,i]
	else:
		for i in range(n): # i ranging from 0 to n-1
			x[i] /= R[i,i]
			for j in range(i+1,n): # j ranging from i+1 to n-1
				x[j] -= x[i] * R[j,i]
	return x


def MGS(A, b = None):
	# reduced QR 
	m,n = A.shape
	Q = np.copy(A)
	R = np.zeros((n,n))

	#vectorize
	for i in range(n):
		R[i,i] = norm_2(Q[:,i])
		Q[:,i] /= R[i,i]
		R[i,i+1:] = Q[:,i+1:].T @ Q[:,i]  
		Q[:,i+1:] = Q[:,i+1:] - np.outer(Q[:,i],R[i,i+1:])

	if (b is not None):
		x = Q.T @ b
		return backsolve(R,x)

	return Q,R


def HouseholderQR(A, b = None, reduced=True):
	## Assuming A is m*n where m>=n
	## if reduced, Q.shape = mxn, R.shape = nxn, 
	## else, 	   Q.shape = mxm, R.shape = mxn, 
	m,n = A.shape
	V_list = []
	if b is None:
		# find R
		R = np.copy(A)
		for i in range(n):
			v = R[i:,i].copy()
			v[0] = v[0] + (2*(v[0]>0)-1) * norm_2(v)
			v = v / norm_2(v)
			V_list.append(v)
			R[i:,i:] = R[i:,i:] - 2 * np.outer(v , (v.T @ R[i:,i:]))

		# check if reduced form
		if reduced==True:
			R = R[0:n,0:n]
			Q = np.append(np.eye(n),np.zeros((m-n,n)),axis=0)
		else:
			Q = np.eye(m)

		# form Q
		for k in range(n):
			Q[n-1-k:,:] = Q[n-1-k:,:] - 2*np.outer(V_list[n-1-k],(V_list[n-1-k] @ Q[n-1-k:,:]))

		return Q,R

	else:
		x = b.copy()
		R = np.copy(A)
		for i in range(n):
			v = R[i:,i].copy()
			v[0] = v[0] + (2*(v[0]>0)-1) * np.linalg.norm(v,2)
			v = v / np.linalg.norm(v,2)
			x[i:] = x[i:] - 2 * v * (v @ x[i:])
			R[i:,i:] = R[i:,i:] - 2 * np.outer(v , (v.T @ R[i:,i:]))
		return backsolve(R,x)



def GivensQR(A, b = None):
	# It seems GivensQR cannot form reduced QR

	m,n = A.shape

	if b is None:
		Q = np.eye(m)
		R = np.copy(A)

		for j in range(n):
			for i in range(m-1,j,-1): # from m-1 to j
				c = R[i-1,j] / np.sqrt(R[i-1,j]**2 + R[i,j]**2)
				s = -R[i,j]  / np.sqrt(R[i-1,j]**2 + R[i,j]**2)

				# The following is to form G explicitely, easy but too costly
				# G = np.eye(m)
				# G[i-1,i-1]=c
				# G[i,i]=c
				# G[i-1,i]=-s
				# G[i,i-1]=s
				# R=G@R
				# Q=Q@G.T

				G_22 = np.array([[c,-s],[s,c]])
				R[i-1:i+1,j:]= G_22 @ R[i-1:i+1,j:] # The second indice is starting from j as R is partially upper triangular by construction
				Q[:,i-1:i+1] = Q[:,i-1:i+1]   @ G_22.T

		return Q,R

	else:
		x = b.copy()
		R = np.copy(A)
		for j in range(n):
			for i in range(m-1,j,-1): # from m-1 to j
				c = R[i-1,j] / np.sqrt(R[i-1,j]**2 + R[i,j]**2)
				s = -R[i,j]  / np.sqrt(R[i-1,j]**2 + R[i,j]**2)
				G_22 = np.array([[c,-s],[s,c]])
				R[i-1:i+1,j:]= G_22 @ R[i-1:i+1,j:] 
				x[i-1:i+1]	 = G_22 @ x[i-1:i+1]

		return backsolve(R,x)




if __name__ == '__main__':

	m = 50
	n = 50	
	QR_method={}
	QR_method['C1'] = MGS
	QR_method['C2'] = HouseholderQR
	QR_method['C3'] = GivensQR

	for method in QR_method:
		QR = QR_method[method]
		print("Begin Problem " + method," : ",QR.__name__)
		#################### Part(a)########################
		errors = [0.0,0.0]
		for i in range(100):
			A = np.random.random((m, n))
			Q,R = QR(A)
			errors[0]+= np.linalg.norm(Q.T @ Q - np.eye(n),2)
			errors[1]+= np.linalg.norm(Q @ R - A,2)
		errors[0]/=100
		errors[1]/=100
		print("The average values of |QˆT Q − I| is: {:.2e}".format(errors[0]))
		print("The average values of |QR − A|    is: {:.2e}".format(errors[1]))


		#################### Part(b)########################
		errors = 0.0
		for i in range(100):
			A = np.random.random((m, n))
			x = np.random.random(n)
			b = A @ x
			x_calculated = QR(A,b)
			errors+= np.linalg.norm(x-x_calculated,2)
		errors/=100
		print("The average values of |x - x_c|   is: {:.2e}.\n".format(errors))
	exit(0)








