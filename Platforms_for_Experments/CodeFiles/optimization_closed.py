## Contains All Helper Functions for Optimization
import numpy as np

#Calculate value of objective function(vectorized case)
def objective_function_vectorized(y: np.ndarray, X: np.ndarray, w: np.ndarray, alpha, b = None):
    I = (X @ w).flatten()
    y = y.flatten()
    w = w.flatten()

    if b is not None:
      b = b.flatten()
      function = (np.linalg.norm(y - I - b) ** 2) + (alpha * (np.linalg.norm(w) ** 2))
    else:
      function = (np.linalg.norm(y - I) ** 2) + (alpha * (np.linalg.norm(w) ** 2))
    return function

#Calculate value of objective function(tensor case)
def objective_function_tensor(y: np.ndarray, X: np.ndarray, B: np.ndarray, alpha,b = None):
    I = inner_product(X, B).flatten()
    y = y.flatten()
    B = B.flatten()
    if b is not None:
      b = b.flatten()
      function = (np.linalg.norm(y - I -b) ** 2) + (alpha * (np.linalg.norm(B) ** 2))
    else:
      function = (np.linalg.norm(y - I) ** 2) + (alpha * (np.linalg.norm(B) ** 2))
    return function

#Calculate the objective function value with the separable regularizing term
def objective_function_tensor_sep(y: np.ndarray, X: np.ndarray, B: np.ndarray,lsr_ten, alpha,b = None):
    I = inner_product(X, B).flatten()
    y = y.flatten()
    B = B.flatten()
    regularizer = 0
    print('Separable Function')
    #developing the separable regularizing term
   
    separation = len(lsr_ten.factor_matrices)
    tucker = len(lsr_ten.factor_matrices[0])
    
    for s in range(separation):
       for k in range(tucker):
          regularizer += (np.linalg.norm(lsr_ten.factor_matrices[s][k])**2)
    regularizer = regularizer + (np.linalg.norm(lsr_ten.core_tensor)**2)

    if b is not None:
      b = b.flatten()
      function = (np.linalg.norm(y - I -b) ** 2) + (alpha * regularizer)
    else:
      function = (np.linalg.norm(y - I) ** 2) + (alpha * regularizer)
    return function


#Calculate x* and p* for Objective Function(Tensor Case)
#X_train is a Tensor of samples x m x n
#Y_train is a normal vector of size samples x 1. It can also be a flattened array of size (samples, )
def calculate_optimal_iterate_and_function_value(X_train: np.ndarray, Y_train: np.ndarray, lambda1):
    X_train = X_train.reshape((X_train.shape[0], -1))
    Y_train = Y_train.reshape((-1, 1))

    #Calculate Optimal Weight Tensor and Optimal Objective Function Value
    B_optimal = np.linalg.inv(X_train.T @ X_train + lambda1 * np.eye(X_train.shape[1])) @ X_train.T @ Y_train
    I = X_train @ B_optimal
    p_star = (np.linalg.norm(Y_train - I) ** 2) + (lambda1 * (np.linalg.norm(B_optimal) ** 2))

    return B_optimal, p_star


#Inner product of two tensors
#tensor1: samples x m x n
#tensor2: m x n
def inner_product(tensor1: np.ndarray, tensor2: np.ndarray):
    tensor1 = tensor1.reshape(tensor1.shape[0], -1)
    tensor2 = tensor2.reshape(-1, 1)
    return tensor1 @ tensor2

#Calculate R2 Score
def R2(y_true, y_pred):
    #Flatten for insurance
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    #Calculate R2 Score
    y_true_mean = np.mean(y_true)
    tss = np.sum((y_true - y_true_mean) ** 2)
    rss = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (rss / tss)
    return r2