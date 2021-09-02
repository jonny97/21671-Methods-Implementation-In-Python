import numpy as np
np.set_printoptions(suppress=True)
QR_lib     = __import__('QR_lib')

def StOMP(A,y):
    # t is the threshold to pick indices
    # solve underdetermined system Ax = y, with m<N, return x in R^N
    m,N = A.shape
    S     = []
    x     = np.zeros(N)
    r0    = A.T@y


    for i in range(4):
        r = A.T@(y-A@x)
        if np.max(np.abs(r))<np.max(np.abs(r0))*0.005 or i*i>m: 
            # stop if r is small enough or i*i>m
            break
        t = np.max(np.abs(r))*0.1
        for idx in range(N):
            if np.abs(r[idx])>t:
                if idx not in S:
                    S.append(idx)

        ### now solve R.T @ R @ x[S] = Q @ R @ y : x[S] = R^-1 @ Q.T @ y  
        [Q,R] = QR_lib.HouseholderQR(A[:,S])
        Q_T_y = Q.T @ y
        x[S] = QR_lib.backsolve(R,Q_T_y)

    print("S,x: \n",S,x[S],"\n")


    return x






