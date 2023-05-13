# Este trabalho é referente à Task02 de Algebra Linear Computacional
# e implementa métodos para cálculo de autovetores e autovalores

'''
Métodos para obtenção de autovetores e autovalores
'''

import numpy as np

class AlgLinMethods():
    '''
        Contém métodos para cálculo de autovalores e autovetores de uma matriz, 
        e um método extra para resolução de um sistema linear através destes 
        autovalores e autovetores.

        Parâmentos:
        -----------
        matrix : array-like
            Matriz para análise.
    '''
    def __init__(self, matrix):
        if not isinstance (matrix, (np.ndarray, list, tuple)):
            raise TypeError("A matriz de coeficientes deve ser um objeto array-like.")
        matrix = np.array(matrix, dtype=np.float64)
        
        # Verifica se a matriz dos coeficientes é quadrada
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("A matriz dos coeficientes não é quadrada.")

        self.matrix = matrix
        self.n = matrix.shape[0] # Numero de linhas/colunas da matriz
    
    def powerMethod(self, tol=1e-3, max_iter = 100):
        """
        Calcula o maior autovalore e correspondente autovetor de uma matriz 
        usando o método da Potência.

        Parâmetros:
        -----------
        tol : float
            Tolerância para o critério de parada (padrão: 1e-3).
        max_iter : int
            Número máximo de iterações (padrão: 100).

        Retorna:
        --------
        lambda_ : float
            O maior autovalor (absoluto) da matriz.
        autovetores : array-like
            O autovetor correspondente ao maior autovalor.
        """
        # Inicialização
        x = np.ones(self.n) # Autovetor inicial
        lambda_ant = 1 # Autovalor inicial
        error = np.inf
        count_iter = 0

        while error > tol and count_iter < max_iter:
            Ax = np.dot(self.matrix, x)
            lambda_ = np.linalg.norm(Ax, np.inf) # Autovalor estimado
            x = Ax / lambda_

            error = np.abs(lambda_ - lambda_ant) / np.abs(lambda_)
            lambda_ant = lambda_
            count_iter += 1
        
        if count_iter == max_iter:
            print("Número máximo de iterações atingido. A solução pode ser imprecisa.")

        return lambda_, x
    
    def jacobiMethod(self, tol=1e-3, max_iter = 100):
        """
        Calcula autovalores e autovetores de uma matriz usando o método de Jacobi.

        Parâmetros:
        -----------
        tol : float
            Tolerância para o critério de parada (padrão: 1e-3).
        max_iter : int
            Número máximo de iterações (padrão: 100).

        Retorna:
        --------
        autovalores : array-like
            Os autovalores da matriz.
        autovetores : array-like
            Os autovetores correspondentes aos autovalores.
        """
        # Inicialização
        A = np.copy(self.matrix) 
        n = self.n
        V = np.eye(n)
        count_iter = 0

        # Verifica se a matriz é simétrica
        if not np.allclose(A, A.T, atol=tol):
            raise ValueError("A matriz não é simétrica.")

        # Iterações de Jacobi
        while count_iter < max_iter:
            # Encontra o índice (i,j) do elemento máximo fora da diagonal
            idx_max = np.argmax(np.abs(np.triu(A, k=1)))
            i, j = np.unravel_index(idx_max, (n, n))

            # Verifica se a tolerância foi atingida
            if np.abs(A[i, j]) < tol:
                break

            # Calcula o ângulo de rotação
            if A[i, i] == A[j, j]:
                theta = np.pi/4
            else:
                theta = 0.5 * np.arctan2(2*A[i, j], A[i, i] - A[j, j])

            # Calcula a matriz de rotação
            R = np.eye(n)
            R[i, i] = R[j, j] = np.cos(theta)
            R[i, j] = -np.sin(theta)
            R[j, i] = np.sin(theta)

            # Atualiza a matriz A e a matriz de autovetores V
            A = np.dot(R.T, np.dot(A, R))
            V = np.dot(V, R)

            count_iter += 1

        if count_iter == max_iter:
            print("Número máximo de iterações atingido. A solução pode ser imprecisa.")

        return A, V
    
    def det(self, eigenvalues):
        """
        Calcula o determinante da matriz através de seus autovalores.

        Parâmetros:
        -----------
        eigenvalues : array-like
            Matriz com os autovalores da matriz dos coeficientes.

        Retorna:
        --------
        det : float
            O determinante da matriz.
        """
        det = np.prod(np.diag(eigenvalues))
        return det
    
if __name__ == '__main__':
    # TESTES
    m1 = np.array([[1, 0.2, 0], 
                  [0.2, 1, 0.5], 
                  [0, 0.5, 1]], dtype=np.float64)
    
    m2 = np.array([[9, 6, 0], 
                  [6, 9, -3], 
                  [0, -3, 9]], dtype=np.float64)

    f = AlgLinMethods(m1)
    print(f.jacobiMethod())
