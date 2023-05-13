'''
Programa: main.py
Descrição: Realiza operações de álgebra linear de acordo com o requisitado pelo usuário. 
        A matriz e os vetores usados nas computações estão disponíveis na pasta 'Test_Files'.
Autor: Bárbara Cerqueira
Data: 12/05/2023
'''

from Task01.LinearSystemSolver import LinearSystemSolver
from Task02.AlgLinMethods import AlgLinMethods
import numpy as np
import sys

def main():

    # Importa as matrizes e vetores da pasta Test_Files
    try:
        matrizA = np.loadtxt('Test_Files/Matriz_A.dat')
        vetorb1 = np.loadtxt('Test_Files/Vetor_B_01.dat')
        vetorb2 = np.loadtxt('Test_Files/Vetor_B_02.dat')
        vetorb3 = np.loadtxt('Test_Files/Vetor_B_03.dat')
    except Exception as e:
        print(f'Erro ao tentar ler os arquivos da matriz/vetores: {e}')
        while(True): 
            input('\nPressione qualquer tecla para encerrar...')
            sys.exit(1)

    # Inicializa as classes necessárias
    try:
        M = LinearSystemSolver(matrizA)
        E = AlgLinMethods(matrizA)
    except Exception as e:
        print(f'Erro ao tentar inicializar objetos: {e}') 
        while(True): 
            input('\nPressione qualquer tecla para encerrar...')
            sys.exit(1)

    while(True):
        print("\nAlgebra Linear Computacional\n"
            "----------------------------\n"
            "Operações disponíveis:\n"
            "(1) Resolução de sistema linear\n"
            "(2) Autovalores e autovetores\n"
            "(3) Sair\n")
        
        # Recebe e verifica o código principal
        while (True):
            tcod = input("Insira o código da operação a ser executada: ")
            try:
                tcod = int(tcod)
                if tcod not in (1, 2, 3): raise ValueError
            except ValueError:
                print("Código Inválido!")
                continue
            else:
                break

        if tcod == 1:  # Resolução de Sistemas Lineares
            print("\nOperações disponíveis:\n"
            "(1) Decomposição LU\n"
            "(2) Decomposição de Cholesky\n"
            "(3) Procedimento iterativo Jacobi\n"
            "(4) Procedimento iterativo Gauss-Seidel\n"
            "(5) Retornar\n"
            "(6) Sair\n")

            # Recebe e verifica o código de nível secundário
            while (True):
                icod = input("Insira o código do método para solução de sistemas lineares: ")
                try:
                    icod = int(icod)
                    if icod not in (1, 2, 3, 4, 5, 6): raise ValueError
                except ValueError:
                    print("Código Inválido!")
                    continue
                else:
                    break

            if icod == 1:
                print("\nResolvendo o sistema A*x_i = b_i com Decomposição LU: \n")
                print(f"Solução de A*x_1 = b_1:")
                print(f"{M.solve(vetorb1, method='LU')}\n")
                print(f"Solução de A*x_2 = b_2:")
                print(f"{M.solve(vetorb2, method='LU')}\n")
                print(f"Solução de A*x_3 = b_3:")
                print(f"{M.solve(vetorb3, method='LU')}\n")

            elif icod == 2:
                print("\nResolvendo o sistema A*x_i = b_i com Decomposição de Cholesky: \n")
                print(f"Solução de A*x_1 = b_1:")
                print(f"{M.solve(vetorb1, method='Cholesky')}\n")
                print(f"Solução de A*x_2 = b_2:")
                print(f"{M.solve(vetorb2, method='Cholesky')}\n")
                print(f"Solução de A*x_3 = b_3:")
                print(f"{M.solve(vetorb3, method='Cholesky')}\n")

            elif icod == 3:
                print("\nResolvendo o sistema A*x_i = b_i com o Procedimento iterativo de Jacobi: \n")
                print(f"Solução de A*x_1 = b_1:")
                print(f"{M.solve(vetorb1, method='Jacobi', tol=1e-7, max_iter=5000)}\n")
                print(f"Solução de A*x_2 = b_2:")
                print(f"{M.solve(vetorb2, method='Jacobi', tol=1e-7, max_iter=5000)}\n")
                print(f"Solução de A*x_3 = b_3:")
                print(f"{M.solve(vetorb3, method='Jacobi', tol=1e-7, max_iter=5000)}\n")
            
            elif icod == 4:
                print("\nResolvendo o sistema A*x_i = b_i com o Procedimento iterativo de Gauss Seidel: \n")
                print(f"Solução de A*x_1 = b_1:")
                print(f"{M.solve(vetorb1, method='GaussSeidel', tol=1e-7, max_iter=5000)}\n")
                print(f"Solução de A*x_2 = b_2:")
                print(f"{M.solve(vetorb2, method='GaussSeidel', tol=1e-7, max_iter=5000)}\n")
                print(f"Solução de A*x_3 = b_3:")
                print(f"{M.solve(vetorb3, method='GaussSeidel', tol=1e-7, max_iter=5000)}\n")

            elif icod == 5:
                continue
            else:
                sys.exit(0)
            input("Pressione Enter para voltar ao início...")
            
        elif tcod == 2: # Autovalores e Autovetores
            print("\nOperações disponíveis:\n"
            "(1) Método da Potência\n"
            "(2) Método de Jacobi\n"
            "(3) Retornar\n"
            "(4) Sair\n")

            # Recebe e verifica o código de nível secundário
            while (True):
                icod = input("Insira o código do método para descobrir autovalores e autovetores: ")
                try:
                    icod = int(icod)
                    if icod not in (1, 2, 3, 4): raise ValueError
                except ValueError:
                    print("Código Inválido!")
                    continue
                else:
                    break

            if icod == 1:
                print("\nDescobrindo os autovalores e autovetores da matriz A através do Método da Potência: \n")
                print("Warning: O Método da Potência só é capaz de encontrar o maior autovalor absoluto e seu autovetor associado. "
                      "Portanto, não será possível obter o determinante da matriz A através dele.\n")
                autovalor, autovetor = E.powerMethod(tol=1e-7, max_iter=5000)
                print(f"Maior autovalor: {autovalor}")
                print(f"Autovetor associado: \n{autovetor}\n")

            elif icod == 2:
                print("\nDescobrindo os autovalores e autovetores da matriz A através do Método de Jacobi: \n")
                autovalor, autovetor = E.jacobiMethod(tol=1e-7, max_iter=5000)
                print(f"Autovalores: {np.diag(autovalor)}")
                print(f"Autovetores: \n{autovetor}")
                print(f"Determinante de A: {E.det(autovalor)}\n")
                
            elif icod == 3:
                continue
            else:
                sys.exit(0)
            input("Pressione Enter para voltar ao início...")

        else:
            sys.exit(0)

main()

