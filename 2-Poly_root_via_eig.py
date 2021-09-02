import numpy as np


def Roots_Polynomial(coefficients):
	# Roots_Polynomial: assume coefficients are [a_0,a_1,...,a_m-1,a_m]
	m = len(coefficients)-1	
	# handle zeros at the end of the sequence
	while coefficients[m]==0:
		m-=1
	# normalize so that the last entry is 1
	if (coefficients[m]!=1):
		coefficients /= coefficients[m]
	# construct the matrix whose eigenvalues are polynomial roots
	A = np.diag(np.ones(m-1),-1)
	A[:,-1] = coefficients[:m]	
	A[:,-1] *=-1		
	return np.linalg.eigvals(A)

'''
	Note: given the error is eigenvalue finding algorithms, 
	it might be better to further do some fix point iterations(Newton) to get higher precision
'''

print(Roots_Polynomial(np.array([-0.57,0.114,0.0355/2.0])))
print(Roots_Polynomial(np.array([-0.57,0.15,0.0355/2.0])))
print(Roots_Polynomial(np.array([-0.57,0.19,0.0355/2.0])))
print(Roots_Polynomial(np.array([-0.57,0.26,0.0355/2.0])))
