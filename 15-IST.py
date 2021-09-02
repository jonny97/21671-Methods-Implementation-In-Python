import numpy as np
np.set_printoptions(suppress=True)
QR_lib     = __import__('QR_lib')

def S(x,tao):
    ret = np.zeros(x.shape)
    for i in range(len(x)):
        if x[i]==0:
            continue
        else:
            ret[i] = x[i]*max(0,1-tao/np.abs(x[i]))
    return ret

def IST(A,y,s,tao=2,lambda_ = 0.1):
    # t is the threshold to pick indices
    # solve underdetermined system Ax = y, with m<N, return x in R^N
    m,N = A.shape
    x     = np.zeros(N)
    r0    = A.T@y
    prev_r= np.zeros(r0.shape)


    for i in range(m):
        r = r0-A.T@(A@x)
        #print(np.linalg.norm(prev_r),np.linalg.norm(r))
        if (np.linalg.norm(prev_r-r)<np.linalg.norm(r)*10e-10):
            break
        else:
            prev_r = r
        x = S(x+tao*r,lambda_*tao)

    return x






