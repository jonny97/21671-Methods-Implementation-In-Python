import numpy as np
np.set_printoptions(suppress=True)

SVD = __import__('6-SVD_Phase2')
QR     = __import__('QR_lib')


def SVD_randomized(A,k):
    n = A.shape[1]

    G = np.random.randn(n,k)

    Y = A @ G

    Q,R = QR.HouseholderQR(Y)
    B = Q.T @ A

    U_tilde,S,V = SVD.DemmelKahan(B)

    U = Q@U_tilde

    return U,S,V



if __name__ == "__main__":
    m,n=100,100;
    k  = 25
    np.random.seed(0)
    A = np.random.random((m,n))
    Q1,R,Q2 = SVD_randomized(A,k)

    print(np.max(np.abs(Q1.T@Q1-np.eye(k))))
    print(np.max(np.abs(Q2.T@Q2-np.eye(k))))
    exact_k_singular_value = -np.sort(-np.linalg.svd(A)[1])[:k]
    print("computed singular values: \n",np.diag(R))
    print("exact singular values: \n",exact_k_singular_value)
    print("relative error for singular values: \n",np.abs(np.diag(R)-exact_k_singular_value)/exact_k_singular_value)
    print("2-norm of Q1@R@Q2.T-A",np.linalg.norm(Q1@R@Q2.T-A,2))
    print("2-norm of (Q1@R@Q2.T-A)/sigma_(k+1)",np.linalg.norm(Q1@R@Q2.T-A,2)/-np.sort(-np.linalg.svd(A)[1])[k])


    # ek = []
    # for k in range(0,99,1):
    #     Q1,R,Q2 = SVD_randomized(A,k)
    #     print(k,"2-norm of (Q1@R@Q2.T-A)/sigma_(k+1)",np.linalg.norm(Q1@R@Q2.T-A,2)/-np.sort(-np.linalg.svd(A)[1])[k])
    #     ek.append(np.linalg.norm(Q1@R@Q2.T-A,2)/-np.sort(-np.linalg.svd(A)[1])[k])

    # print(ek)


