import matplotlib.pyplot as plt
import numpy
import scipy
import matplotlib.pyplot as plt

import numpy as np
from scipy.sparse import csr_matrix
from scipy.io import mmread
import matplotlib.pyplot as plt

def convergence_analysis(errors, method_name):
    final_error = errors[-1]
    convergence_rate = (np.log(errors[-1]) - np.log(errors[-2])) / (np.log(errors[-2]) - np.log(errors[-3]))
    print(f"Convergence analysis for {method_name}:")
    print(f"Final error: {final_error}")
    print(f"Convergence rate: {convergence_rate}\n")

def plot_convergence(method_name, errors):
    iterations = range(len(errors))
    plt.plot(iterations, np.log(errors), label=method_name)


def is_diagonally_dominant(A):
    diags = np.abs(A.diagonal())
    sums = np.sum(np.abs(A), axis=1) - diags
    return np.all(diags >= sums)

def spectral_radius(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    return np.max(np.abs(eigenvalues))

def jacobi(A, b, x, max_iterations=1000, tol=1e-10):
    D = np.diag(A)
    R = A - np.diagflat(D)
    for i in range(max_iterations):
        x_new = (b - np.dot(R, x)) / D
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x

def gauss_seidel(A, b, x, max_iterations=1000, tol=1e-10):
    L = np.tril(A)
    U = A - L
    for i in range(max_iterations):
        x_new = np.linalg.solve(L, b - np.dot(U, x))
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x

def sor(A, b, x, omega, max_iterations=1000, tol=1e-10):
    L = np.tril(A)
    U = A - L
    for i in range(max_iterations):
        x_new = np.linalg.solve(L, b - np.dot(U, x))
        x = x + omega * (x_new - x)
        if np.linalg.norm(x_new - x) < tol:
            break
    return x

def is_diagonally_dominant(A):
    diags = np.abs(A.diagonal())
    sums = np.sum(np.abs(A), axis=1) - diags
    return np.all(diags >= sums)


def metodo_iterativos(S):
	A = S.todense(order='C')
	n = A.shape[0]
	b = A.dot(np.ones((n, 1)))

	diagonally_dominant = is_diagonally_dominant(A)

	spectral_radius_value = spectral_radius(A)

	x = np.zeros(n)

	x_jacobi = jacobi(A, b, x)
	x_gauss_seidel = gauss_seidel(A, b, x)
	omega = 1.2  # Valor de omega escolhido para o SOR
	x_sor = sor(A, b, x, omega)

	x = np.zeros(n)

	max_iterations = 1000
	tolerance = 1e-10

	errors_jacobi = []
	errors_gauss_seidel = []
	errors_sor = []

	x_jacobi = x
	x_gauss_seidel = x
	x_sor = x

	for i in range(max_iterations):
	    x_jacobi_new = jacobi(A, b, x_jacobi, max_iterations=1, tol=tolerance)
	    x_gauss_seidel_new = gauss_seidel(A, b, x_gauss_seidel, max_iterations=1, tol=tolerance)
	    x_sor_new = sor(A, b, x_sor, omega, max_iterations=1, tol=tolerance)

	    error_jacobi = np.linalg.norm(A.dot(x_jacobi_new) - b)
	    error_gauss_seidel = np.linalg.norm(A.dot(x_gauss_seidel_new) - b)
	    error_sor = np.linalg.norm(A.dot(x_sor_new) - b)

	    errors_jacobi.append(error_jacobi)
	    errors_gauss_seidel.append(error_gauss_seidel)
	    errors_sor.append(error_sor)

	    x_jacobi = x_jacobi_new
	    x_gauss_seidel = x_gauss_seidel_new
	    x_sor = x_sor_new

	plot_convergence('Jacobi', errors_jacobi)
	plot_convergence('Gauss-Seidel', errors_gauss_seidel)
	plot_convergence(f'SOR (omega={omega})', errors_sor)

	plt.yscale('log')
	plt.xlabel('Iterations')
	plt.ylabel('Log(Error)')
	plt.title('Convergence Comparison of Iterative Methods')
	plt.legend()
	plt.show()


	convergence_analysis(errors_jacobi, "Jacobi")

	convergence_analysis(errors_gauss_seidel, "Gauss-Seidel")

	convergence_analysis(errors_sor, f"SOR (omega={omega})")


S = []
S.append(scipy.io.mmread('10.mtx'))
#S.append(scipy.io.mmread('100.mtx'))
#S.append(scipy.io.mmread('1000.mtx'))
#S.append(scipy.io.mmread('10000.mtx'))
#S.append(scipy.io.mmread('100000.mtx'))

for s in S:
    print("entrando no primeiro metodo")
    metodo_iterativos(s)

