import numpy as np
np.set_printoptions(suppress=True)

def GolubKahan(A, reduced=True):
    # GolubKahan bidiagonalization, decompose A = Q1 @ B @ Q2.T 
    # reduced: if A is MxN,(Q1: mxn, B: nxn, C: nxn)
    # not reduced: (Q1: mxm, B: mxn, C: nxn)
    # return Q1,B,Q2
    m,n = A.shape
    V1_list = []
    V2_list = []
    R = np.copy(A)
    for i in range(n):
        # apply on the left
        v = R[i:,i].copy()
        v[0] = v[0] + (2*(v[0]>0)-1) * np.linalg.norm(v,2)
        v = v / np.linalg.norm(v,2)
        V1_list.append(v)
        R[i:,i:] = R[i:,i:] - 2 * np.outer(v , (v.T @ R[i:,i:]))
        # apply on the left
        if i<n-1:
            v = R[i,i+1:].copy()
            v[0] = v[0] + (2*(v[0]>0)-1) * np.linalg.norm(v,2)
            v = v / np.linalg.norm(v,2)
            V2_list.append(v)
            R[i:,i+1:] = R[i:,i+1:] - 2 * np.outer((R[i:,i+1:] @ v),v)

        
    # check if reduced form
    if reduced==True:
        R = R[0:n,0:n]
        Q1 = np.append(np.eye(n),np.zeros((m-n,n)),axis=0)
    else:
        Q1 = np.eye(m)
    Q2 = np.eye(n)

    
    # form Q
    for k in range(len(V1_list)):
        Q1[n-1-k:,:] = Q1[n-1-k:,:] - 2*np.outer(V1_list[-k-1],(V1_list[-k-1] @ Q1[n-1-k:,:]))

    for k in range(len(V2_list)):
        Q2[n-1-k:,:] = Q2[n-1-k:,:] - 2*np.outer(V2_list[-k-1],(V2_list[-k-1] @ Q2[n-1-k:,:]))

    return Q1,R,Q2


if __name__ == "__main__":
    m,n=8,5;
    A = np.random.random((m,n));
    Q1,R,Q2=GolubKahan(A);
    print(R)
    print(np.max(np.abs(Q1@R@Q2.T-A)))
    print(np.max(np.abs(Q1.T@Q1-np.eye(n))))
    print(np.max(np.abs(Q2.T@Q2-np.eye(n))))
