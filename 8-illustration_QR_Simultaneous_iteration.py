import numpy as np
np.set_printoptions(suppress=True)

n= 6


Hessenberg = __import__('3-Hessenberg_Hermitian_optimized')
QR_lib 	   = __import__('QR_lib')
A = np.random.random((n,n))
A = A+A.T
Q,H=Hessenberg.Hessenberg(A,Hermitian=1)



def QR_method_list(A):
	Q_list,R_list,A_list =([],[],[])
	Q,R = QR_lib.HouseholderQR(A)
	A_list.append(A)
	Q_list.append(Q)
	R_list.append(R)
	for i in range(10):
		A = R@Q
		Q,R = QR_lib.HouseholderQR(A)
		A_list.append(A)
		Q_list.append(Q_list[-1]@Q)
		R_list.append(R@R_list[-1])	
	return Q_list,R_list,A_list 


def Simutaneous_Iteration_list(A):
	Q_list,R_list,A_list =([],[],[])
	Q,R = QR_lib.HouseholderQR(A)
	A_list.append(A)
	Q_list.append(Q)
	R_list.append(R)
	for i in range(10):
		A_list.append(Q.T@A@Q)
		Q,R = QR_lib.HouseholderQR(A@Q)
		Q_list.append(Q)
		R_list.append(R@R_list[-1])	
	return Q_list,R_list,A_list 	


q1,r1,a1 = QR_method_list(A)
q2,r2,a2 = Simutaneous_Iteration_list(A)
print(max([np.max(np.abs(a1[i])-np.abs(a2[i])) for i in range(11)]))
print(max([np.max(np.abs(q1[i])-np.abs(q2[i])) for i in range(11)]))
print(max([np.max(np.abs(r1[i])-np.abs(r2[i])) for i in range(11)]))



def QR_method(B):
	A = np.copy(B)
	for i in range(30):
		Q,R = QR_lib.HouseholderQR(A)
		A = R@Q
		print(i)
		print(A)
	return A

def Simutaneous_Iteration(A):
	Q,R = QR_lib.HouseholderQR(A)
	#T   = A@A@A@A 
	# this idea will be bad as we will have A as a tridiagonal matrix, 
	# power will lose such pattern
	for i in range(30):
		Q,R = QR_lib.HouseholderQR(A@Q)
		print(i)
		print(Q.T@A@Q)
	return Q.T@A@Q

#QR_method(A)
Simutaneous_Iteration(A)

print("-------------------------")
Q,H=Hessenberg.Hessenberg(A,Hermitian=1)
Simutaneous_Iteration(H)
