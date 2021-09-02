import numpy as np
np.set_printoptions(suppress=True)
QR_lib     = __import__('QR_lib')

def OMP(A,y):
    # solve underdetermined system Ax = y, with m<N, return x in R^N
    m,N = A.shape
    S     = []
    x     = np.zeros(N)
    r0    = A.T@y
    c     = np.zeros((0,))  # c keeps track of A[:,S].T @ y
    ATA_S = np.zeros((N,0)) # ATA_S keeps track of A.T @ A[:,S]
    Q     = np.zeros((m,0))
    R     = np.zeros((0,0))

    for i in range(10):
        r = r0 - ATA_S@x[S]

        if np.max(np.abs(r))<np.max(np.abs(r0))*0.005 or i*i>m: 
            # stop if r is small enough or i*i>m
            break

        idx = np.argmax(np.abs(r))
        S.append(idx)

        new_Q = A[:,idx] - Q@(Q.T@A[:,idx])
        norm  = np.linalg.norm(new_Q,2)

        R = np.c_[R,Q.T@A[:,idx]]
        R = np.r_[R,np.zeros((1,i+1))]
        R[i,i] = norm
        Q = np.c_[Q,new_Q/norm]

        ### now solve R.T @ R @ x[S] = Q @ R @ y : x[S] = inv(A[:,S].T@A[:,S]) @ A[:,S].T @ y
        c   = np.r_[c,np.dot(A[:,idx],y)]
        temp= QR_lib.backsolve(R.T,c,upper=False)
        x[S]= QR_lib.backsolve(R,temp)

        ATA_S = np.c_[ ATA_S, A.T@A[:,idx]]
    #print("S,x: \n",S,x[S],"\n")


    return x





def OMP_naive(A,y):
    # solve underdetermined system Ax = y, with m<N, return x in R^N
    m,N = A.shape
    S = []
    x = np.zeros(N)

    for i in range(4):
        print(i,x)
        print("residue: ",np.max(np.abs(A.T@(y-A@x))))
        print("residue: ",(np.abs(A.T@(y-A@x))))

        idx = np.argmax(np.abs(A.T@(y-A@x)))
        S.append(idx)
        print("S",S)
        x[S]   = np.linalg.inv(A[:,S].T@A[:,S]) @ A[:,S].T @ y


    return x
