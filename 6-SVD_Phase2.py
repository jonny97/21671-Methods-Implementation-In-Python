import numpy as np
np.set_printoptions(suppress=True)

Phase1 = __import__('5-SVD_Phase1')
QR     = __import__('QR_lib')
tol    = 10**-10
MAX_ITER = 100000

def DK_Rotations(e,d):
    # return c,s,r
    if e==0:
        return 0,1,d

    if abs(e)>abs(d):
        tao = d/e
        t   = (1+tao**2)**0.5
        c   = 1/t
        s   = tao *c
        r   = t*e
        return c,s,r

    if abs(e)<abs(d):
        tao = e/d
        t   = (1+tao**2)**0.5
        s   = 1/t
        c   = tao *s
        r   = t*d
        return c,s,r     

def Givens_Eliminate_naive(Q1,R,Q2,i,i_start,i_end):
    # eliminate R(i,i+1) via givens rotations
    # then eliminate R(i+1,i) via givens rotations

    # apply Q on the right, Q1 stays the same
    c,s,r = DK_Rotations(R[i,i],R[i,i+1])
    G_22 = np.array([[c,-s],[s,c]])
    R[:,i:i+2]= R[:,i:i+2] @ G_22
    Q2[:,i:i+2]= Q2[:,i:i+2] @ G_22

    # apply Q on the left, Q2 stays the same
    c,s,r = DK_Rotations(R[i,i],R[i+1,i])
    G_22 = np.array([[c,s],[-s,c]])
    R[i:i+2,:]= G_22 @ R[i:i+2,:] 
    Q1[:,i:i+2]= Q1[:,i:i+2] @ G_22.T

    return Q1,R,Q2

def Givens_Eliminate_naive_2(Q1,R,Q2,i,i_start,i_end):
    # eliminate R(i,i+1) via givens rotations
    # then eliminate R(i+1,i) via givens rotations
    # apply Q on the right, Q1 stays the same
    c,s,r = DK_Rotations(R[i,i],R[i,i+1])
    G_22 = np.array([[c,-s],[s,c]])

    # apply DK rotation to R instead of G_22
    if i>i_start:
        R[i-1,i]   = R[i-1,i]/R[i,i]*r 
        R[i-1,i+1] = 0
    R[i,i]     = r
    R[i,i+1]   = 0
    R[i+1,i]   = s * R[i+1,i+1]
    R[i+1,i+1] = c * R[i+1,i+1]

    Q2[:,i:i+2]= Q2[:,i:i+2] @ G_22

    # apply Q on the left, Q2 stays the same
    c,s,r = DK_Rotations(R[i,i],R[i+1,i])
    # apply DK rotation instead of G_22
    R[i,i]   = r
    R[i+1,i] = 0

    G_22 = np.array([[c,s],[-s,c]])
    R[i:i+2,i+1:i+3]= G_22 @ R[i:i+2,i+1:i+3] 
    Q1[:,i:i+2]= Q1[:,i:i+2] @ G_22.T

    return Q1,R,Q2

def Givens_Eliminate_optimal(Q1,R,Q2,i,i_start,i_end):
    # eliminate R(i,i+1) via givens rotations
    # then eliminate R(i+1,i) via givens rotations

    c,s,r = DK_Rotations(R[i,i],R[i,i+1])
    # apply DK rotation to R instead of G_22
    if i>i_start:
        # here R[i-1,i] stores the previous s and R[i-1,i+1] stores the previous c
        R[i-1,i]   = R[i-1,i]*r 
        R[i,i]     = R[i-1,i+1]*r    
        R[i-1,i+1] = 0
    else:
        R[i,i]     = r
    R[i,i+1]   = 0
    R[i+1,i]   = s * R[i+1,i+1]
    R[i+1,i+1] = c * R[i+1,i+1]

    if Q2 is not None:
        G_22 = np.array([[c,-s],[s,c]])
        Q2[:,i:i+2]= Q2[:,i:i+2] @ G_22

    # apply DK rotation instead of G_22
    c,s,r = DK_Rotations(R[i,i],R[i+1,i])
    R[i,i]   = r
    R[i+1,i] = 0

    if i == i_end-1 :
        R[i,i+1]   = s * R[i+1,i+1] 
        R[i+1,i+1] = c * R[i+1,i+1]
    else:
        R[i,i+1]=s
        R[i,i+2]=c

    if Q1 is not None:
        G_22 = np.array([[c,s],[-s,c]])
        Q1[:,i:i+2]= Q1[:,i:i+2] @ G_22.T

    return Q1,R,Q2


def DemmelKahan(A, reduced=True):
    m,n = A.shape
    transpose = False
    if m<n:
        transpose = True
        A=A.T
    Q1,R,Q2 = Phase1.GolubKahan(A)
    
    i_start = 0
    i_end   = min(m,n)-1
    for iter in range(MAX_ITER):
        while i_start <= min(m,n)-2 and abs(R[i_start,i_start+1])<tol:
            i_start+=1

        while i_end>=2 and abs(R[i_end-1,i_end])<tol:
            i_end-=1

        #print("begin iter {:<3}, i_start is {:<3}, i_end is {:<3}".format(iter,i_start,i_end))

        if i_start >= i_end:
            break;

        for i in range(i_start,i_end):
            Q1,R,Q2 = Givens_Eliminate_optimal(Q1,R,Q2,i,i_start,i_end)  

    # make all singular values positive        
    temp = np.diag(np.sign(np.diag(R)))
    R = R@temp
    Q2=Q2@temp

    if transpose:
        return Q2,R,Q1

    return Q1,R,Q2

    


if __name__ == "__main__":
    m,n=100,100;
    #np.random.seed(50)
    A = np.random.random((m,n));

    # make A with specific singular values 
    # Q1,_= QR.HouseholderQR(A)
    # A = np.random.random((n,n));
    # Q2,_= QR.HouseholderQR(A)
    # diag = [100,1,10**-2,10**-6,10**-12,10**-20]
    # A = Q1@np.diag(diag)@Q2.T

    Q1,R,Q2=DemmelKahan(A);
    #print(R)
    print(np.max(np.abs(Q1@R@Q2.T-A)))
    print(np.max(np.abs(Q1.T@Q1-np.eye(n))))
    print(np.max(np.abs(Q2.T@Q2-np.eye(n))))
    print("error for one singular is: ",np.max(np.abs(np.sort(np.diag(R))-np.sort(np.linalg.svd(A)[1]))/np.sort(np.linalg.svd(A)[1])))

