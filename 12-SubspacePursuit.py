import numpy as np
np.set_printoptions(suppress=True)
QR_lib     = __import__('QR_lib')

def L_s(x,s):
    # s largest index of x
    return np.argsort(-np.abs(x))[:s]

def H_s(x,s):
    # s largest entry of x
    ret = np.zeros(x.shape)
    Ls = L_s(x,s)
    ret[Ls] += x[Ls]
    return ret



def SubspacePursuit(A,y,s):
    # t is the threshold to pick indices
    # solve underdetermined system Ax = y, with m<N, return x in R^N
    m,N = A.shape
    S     = [] # support of x
    x     = np.zeros(N)
    r0    = A.T@y

    for i in range(m):
        r = A.T@(y-A@x)

        if np.max(np.abs(r))<np.max(np.abs(r0))*0.005 or i*i>m: 
            # stop if r is small enough or i*i>m
            break

        for idx in L_s(r,s):
            if idx not in S:
                S.append(idx)

        ### now solve R.T @ R @ x[S] = Q @ R @ y : x[S] = R^-1 @ Q.T @ y  
        [Q,R] = QR_lib.HouseholderQR(A[:,S])
        Q_T_y = Q.T @ y
        u = QR_lib.backsolve(R,Q_T_y)

        x     = np.zeros(N)
        x[S] += H_s(u,s)
        if np.all(list(L_s(x,s))==S):
            # if S no longer updates
            break
        S     = list(L_s(x,s))

    return x






