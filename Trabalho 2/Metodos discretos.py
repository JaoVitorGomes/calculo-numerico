import matplotlib.pyplot as plt
import numpy
import scipy


def metodo_discreto(S):
    # Tranformando a matriz em densa
    A = S.todense(order='C')

    # Obtendo os valores P, L e U
    P, L, U = scipy.linalg.lu(A)


    #Verificando a esparcidade de P, L e U
    print("esparcidade de P: ",plt.spy(P))
    print("esparcidade de L: ",plt.spy(L))
    print("esparcidade de U: ",plt.spy(U))

    # Retornando o numero de não zeros
    SL = scipy.sparse.csr_matrix(L)
    SU = scipy.sparse.csr_matrix(U)

    nnza = S.count_nonzero()
    nnzl = SL.count_nonzero()
    nnzu = SU.count_nonzero()

    #print(nnza,nnzl,nnzu)

    # Taxa de preenchimento
    tp = 100 - (nnza/(nnzl+nnzu)) * 100

    #print(tp)


    # Solucionando o sistema linear
    n = A.shape[0]
    #print(n)
    u = numpy.ones((n,1))
    b = A.dot(u)
    x = numpy.linalg.solve(A,b)
    teste = A @ x
    #print(A @ x)
    #print(b)

    if teste.all() == b.all():
        print('é igual')


    # Solucionando o sistema linear esparso
    bs = S.dot(u)
    xs = scipy.sparse.linalg.spsolve(S,bs)

    #print(xs)

    # Distancia relativa 
    dist = numpy.linalg.norm(u - x)/numpy.linalg.norm(u)

    #print(dist)

    # Distancia relativa norm infinita
    distinf = numpy.linalg.norm((A - (P @ L @ U)),numpy.inf)/numpy.linalg.norm(A,numpy.inf)

    #print(distinf)


    # Calculando a norma do residuo
    r = numpy.linalg.norm(b - (A @ x))

    #print(r)

    # Calculando o numero de condicionamento da matriz
    K = numpy.linalg.cond(A)

    #print(K)




# Importando a matrix
S = []
S.append(scipy.io.mmread('10.mtx'))
S.append(scipy.io.mmread('100.mtx'))
S.append(scipy.io.mmread('1000.mtx'))
S.append(scipy.io.mmread('10000.mtx'))
S.append(scipy.io.mmread('100000.mtx'))

for s in S:
    print("entrando no primeiro metodo")
    metodo_discreto(s)




