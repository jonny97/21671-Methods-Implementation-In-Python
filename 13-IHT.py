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


def IHT(A,y,s,mu=0.9):
    # t is the threshold to pick indices
    # solve underdetermined system Ax = y, with m<N, return x in R^N
    m,N = A.shape
    S     = []
    x     = np.zeros(N)
    r0    = A.T@y

    for i in range(m):
        print(i,x)
        r = r0-A.T@(A@x)

        if np.max(np.abs(r))<np.max(np.abs(r0))*0.001 or i*2>m: 
            # stop if r is small enough or i*i>m
            break
        
        x = H_s(x+mu*r,s)


    print("x: \n",x[S],"\n")

    return x






