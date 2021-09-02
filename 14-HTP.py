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


def HTP(A,y,s,mu=0.9):
    # t is the threshold to pick indices
    # solve underdetermined system Ax = y, with m<N, return x in R^N
    m,N = A.shape
    S     = []
    x     = np.zeros(N)
    r0    = A.T@y

    for i in range(m):
        print(i,x,S)
        r = r0-A.T@(A@x)

        if np.max(np.abs(r))<np.max(np.abs(r0))*0.001 or i*2>m: 
            # stop if r is small enough or i*i>m
            break
        
        S = L_s(x+mu*r,s)

        [Q,R] = QR_lib.HouseholderQR(A[:,S])
        Q_T_y = Q.T @ y
        u = QR_lib.backsolve(R,Q_T_y)

        x     = np.zeros(N)
        x[S] += H_s(u,s)
        if np.all(list(L_s(x,s))==S):
            # if S no longer updates
            break
        S     = list(L_s(x,s))


    print("S,x: \n",S,x[S],"\n")

    return x






