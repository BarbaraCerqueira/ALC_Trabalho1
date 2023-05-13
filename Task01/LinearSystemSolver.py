# Este trabalho é referente à Task01 de Algebra Linear Computacional
# e implementa um solucionador de sistemas lineares através
# de diferentes métodos

'''
Métodos para solução de sistemas lineares
'''

import numpy as np

class LinearSystemSolver():
    '''
        Soluciona sistemas lineares através dos métodos Decomposição LU,
        Decomposição de Cholesky, Procedimento iterativo Jacobi e o
        Procedimento iterativo Gauss-Seidel

        Parâmentos:
        -----------
        matrix : array-like
            Matriz de coeficientes.
    '''
    def __init__(self, matrix):
        if not isinstance (matrix, (np.ndarray, list, tuple)):
            raise TypeError("A matriz de coeficientes deve ser um objeto array-like")  
        matrix = np.array(matrix, dtype=np.float64)
        
        # Verifica se a matriz dos coeficientes é quadrada
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("A matriz dos coeficientes não é quadrada.")

        self.matrix = matrix
        self.n = matrix.shape[0] # Grau do Sistema de equações
        self.LU = None # Guarda a matriz de coeficientes decomposta por LU
        self.cholesky = None # Guarda a matriz de coeficientes decomposta por Cholesky
    
    def __decomp_LU(self):
        if self.LU is None:
            self.LU = np.copy(self.matrix)
            for k in range(self.n-1):
                self.LU[k+1:self.n,k] /= self.LU[k,k]
                self.LU[k+1:self.n,k+1:self.n] -= np.outer(self.LU[k+1:self.n,k] , self.LU[k,k+1:self.n])
        else:
            print("A matriz já foi decomposta pelo método LU.")

    def __decomp_Cholesky(self):
        if self.cholesky is None:
            self.cholesky = np.copy(self.matrix)
            for i in range(self.n):
                self.cholesky[i,i] -= np.sum(self.cholesky[i,:i]**2)
                if self.cholesky[i,i] <= 0: # Se o resultado dentro da raiz for negativo, matriz não é positiva definida
                    self.cholesky = None
                    raise TypeError("A matriz não é definida positiva.")
                else:
                    self.cholesky[i,i] = np.sqrt(self.cholesky[i,i])

                self.cholesky[i+1:,i] -= np.dot(self.cholesky[i+1:,:i], self.cholesky[i,:i])
                self.cholesky[i+1:,i] /= self.cholesky[i,i]
        else:
            print("A matriz já foi decomposta pelo método de Cholesky.")

    def _solveLU(self, vector): 
        if self.LU is None:
            self.__decomp_LU()
        Y = np.zeros(self.n)
        for i in range(self.n):
            Y[i] = vector[i] - np.dot(self.LU[i,:i], Y[:i]) 
        X = np.zeros(self.n)
        for i in range(self.n-1, -1, -1):
            X[i] = (Y[i] - np.dot(self.LU[i,i+1:], X[i+1:])) / self.LU[i,i]
        return X
    
    def _solveCholesky(self, vector):
        if self.cholesky is None:
            self.__decomp_Cholesky()
        Y = np.zeros(self.n)
        for i in range(self.n):
            Y[i] = (vector[i] - np.dot(self.cholesky[i,:i], Y[:i])) / self.cholesky[i,i]
        X = np.zeros(self.n)
        for i in range(self.n-1, -1, -1):
            X[i] = (Y[i] - np.dot(self.cholesky[i+1:,i], X[i+1:])) / self.cholesky[i,i]
        return X

    def _solveJacobi(self, vector, tol, max_iter):
        x = np.ones(self.n) # X inicial
        x_new = np.zeros(self.n)
        error = np.inf
        count_iter = 0

        while error > tol and count_iter < max_iter:
            for i in range(self.n):
                s = 0
                for j in range(self.n):
                    if j != i:
                        s += self.matrix[i][j] * x[j]
                x_new[i] = (vector[i] - s) / self.matrix[i][i]

            error = np.linalg.norm(x_new - x)/np.linalg.norm(x_new)
            x = np.copy(x_new)
            count_iter += 1
        
        if count_iter == max_iter:
            print("Número máximo de iterações atingido. A solução pode ser imprecisa.")
            
        return x

    def _solveGaussSeidel(self, vector, tol, max_iter):
        x = np.ones(self.n) # X inicial
        error = np.inf
        count_iter = 0

        while error > tol and count_iter < max_iter:
            x_new = np.copy(x)

            for i in range(self.n):
                s = 0
                for j in range(self.n):
                    if j != i:
                        s += self.matrix[i][j] * x_new[j]
                x_new[i] = (vector[i] - s) / self.matrix[i][i]

            error = np.linalg.norm(x_new - x)/np.linalg.norm(x_new)
            x = np.copy(x_new)
            count_iter += 1
        
        if count_iter == max_iter:
            print("Número máximo de iterações atingido. A solução pode ser imprecisa.")

        return x

    def solve(self, vector, method, tol=1e-3, max_iter=100):
        """
        Resolve um sistema linear Ax=b usando o método selecionado.
        
        Parâmetros:
        -----------
        vector: array-like
            Vetor de constantes do sistema.
        method: string
            Nome do método selecionado. Pode ser 'LU', 'Cholesky', 'Jacobi', 'GaussSeidel'.
        tol : float
            Tolerância para o critério de parada (padrão: 1e-3). Ignorado pelos métodos não-iterativos.
        max_iter : int
            Número máximo de iterações (padrão: 100). Ignorado pelos métodos não-iterativos.

        Retorna:
        --------
        X : array-like
            A solução do sistema linear.
        """
        if not isinstance (vector, (np.ndarray, list, tuple)):
            raise TypeError("O vetor de constantes deve ser um objeto array-like.")
        vector = np.array(vector, dtype=np.float64)
        
        if method == None:
            raise ValueError("É preciso indicar o método de solução do sistema.")

        # Verifica se o vetor de constantes têm dimensões compatíveis com o grau do sistema
        if self.n != vector.shape[0]:
            raise ValueError("O vetor de constantes têm dimensões incompatíveis.")
        
        if method == 'LU':
            X = self._solveLU(vector)
        elif method == 'Cholesky':
            X = self._solveCholesky(vector)
        elif method == 'Jacobi':
            X = self._solveJacobi(vector, tol, max_iter)
        elif method == 'GaussSeidel':
            X = self._solveGaussSeidel(vector, tol, max_iter)
        else:
            raise ValueError("O método selecionado é inválido.")

        return X


if __name__ == '__main__':
    # TESTES
    m = np.array([[1, 2, 2], 
                  [4, 4, 2], 
                  [4, 6, 4]], dtype=np.float64)
    m2 = np.array([[1, 0.2, 0.4], 
                  [0.2, 1, 0.5], 
                  [0.4, 0.5, 1]], dtype=np.float64)

    b = np.array([0.6, -0.3, -0.6], dtype=np.float64)

    f = LinearSystemSolver(m2)
    print(f.solve(b, 'LU'))
    
