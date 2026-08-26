import numpy as np
from abc import ABC, abstractmethod

class AbstractContPhaseVar(ABC): 
    """
    Clase abstracta para definir distribuciiones de tipo Phase-Type continua
    
    Las distribuciones tipo Phase-Type estan caracterizadas por los siguientes parámetros: 
    - α (alpha): Vector de una dimensión de probabilidades iniciales
    - A: Matriz generadora infinitesimal cuadrada de dos dimensiones
        Propiedades
        - La suma de los elementos de cada fila deben ser igual a cero
        - Los elementos de la diagonal principal deben ser negativos
        - Los elementos fuera de la diagonal deben ser no-negativos 
    - n_phases: número de estados (fases) transitorios

    ValueError
        Si A no es cuadrada, si A no es matriz generadora válida,
        o si alpha no tiene dimensión compatible.
    """

    def __init__(self, alpha:np.ndarray, A:np.ndarray): 
      """
      Inicializador de distribución Phase-Type continua  
      """
      "Validación 1: Matriz A cuadrada"
      if A.shape[0] != A.shape[1]: 
         raise ValueError(
            f"Matriz Generadora A debe ser cuadrada"
            f"Dimensiones Actuales Matriz A: {A.shape[0]}x{A.shape[1]}"
         )

      "Número de estados (fases) transitorios"
      self.n_phases = A.shape[0]

      "Validación 2: Alpha debe tener el tamaño correcto"
      self._alpha = np.array(alpha, dtype=float)
      if self._alpha.shape[0] != self.n_phases:
               raise ValueError(
                      f"Vector alpha debe tener tamaño {self.n_phases}. "
                      f"Recibió: {self._alpha.shape[0]}"
                  )


      "Validación 3: Matriz A debe ser una matriz generadora válida"
      self._A = np.array(A, dtype=float)
      if not self._check_sub_generator_matrix(self._A):
            raise ValueError(
                "Matriz A no es una matriz generadora válida:\n"
                "  - Las filas deben sumar 0\n"
                "  - Los elementos diagonales deben ser negativos"
            )

    "Métodos de validación"

    def _check_sub_generator_matrix(self, M:np.ndarray):
            #Check if a given rate matrix has the condition that every row sum <= 0, with at least one < 0 
            check1 = np.max(np.sum(M, axis = 1)) <= 1e-10 and np.min(np.sum(M, axis = 1)) < -1e-10 
            # Check if a given transition matrix has all diagonal elements non positive
            if check1 and np.all(np.diag(M)<0):
                return True
            else:
                return False

    def pdf(self, t:List[float], )
    
    

    




      
