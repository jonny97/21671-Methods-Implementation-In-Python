import numpy as np
np.set_printoptions(suppress=True)


# Phase 1 of eigenvalue decomposition is done in 3-Shur_Hermitian_optimized.py
Phase1 = __import__('3-Shur_Hermitian_optimized')
QR     = __import__('QR_lib')
tol    = 10**-10

def sign(x):
    return (x>0)*2-1

def Eig_Wilkinsin(A,Hermitian = 1):
    # Do eigenvalue decomposition of A
    # note that Wilkinsin scheme only works for Symmetric A, So Hermirian has to be true!
    assert(Hermitian)

    m = A.shape[0]
    Q,H = Phase1.Hessenberg(A,Hermitian = Hermitian)

    N = m # m-N is the number of eigenvalues found, initially it is 0
    iter_count = 0
    while 1:
        while N>=2 and abs(H[N-1,N-2]) < tol:
            N-=1 # deflate
        
        if (N==1):
            break

        delta = (H[N-1,N-1]-H[N-2,N-2])/2
        mu = H[N-1,N-1] - sign(delta) * H[N-1,N-2]**2 / (np.abs(delta) + np.sqrt(delta**2 + H[N-1,N-2]**2))
        iter_count+=1
        print("iter: {: <4} left size: {: <4} current residue is {:.2e} ".format(iter_count,N,H[N-1,N-2]))
        tempQ,R = QR.HouseholderQR(H[:N,:N]-mu*np.eye(N))

        H[:N,:N] = R@tempQ + mu*np.eye(N)
        Q[: ,:N] = Q[: ,:N]@tempQ
    
    return Q,H

test=1
if test:
    n = 100
    A = np.random.random((n,n))
    A = A+A.T
    Q,H = Eig_Wilkinsin(A,1)
    print("\n\n2-norm error between Q.T@Q and I: ",np.linalg.norm(Q.T@Q-np.eye(n),2))
    print("2-norm error between Q@H@Q.T and A: ",np.linalg.norm(Q@H@Q.T-A,2))
    print("max error for one eigenvalue is: ",np.max(np.sort(np.diag(H))-np.sort(np.linalg.eig(A)[0])))


