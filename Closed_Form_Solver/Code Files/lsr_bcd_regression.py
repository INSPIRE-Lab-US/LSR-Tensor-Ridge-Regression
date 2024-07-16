from sklearn.linear_model import Ridge
import LSR_Tensor_2D_v1
import numpy as np
from optimization import objective_function_tensor_sep
from sklearn.linear_model import SGDRegressor
import copy

#lsr_ten: LSR Tensor
#training_data: X
#training_labels: Y
#hypers: hyperparameters

def lsr_bcd_regression(lsr_ten, training_data: np.ndarray, training_labels: np.ndarray, hypers: dict, intercept = False):
    #Get LSR Tensor Information and other hyperparameters
    shape, ranks, sep_rank, order = lsr_ten.shape, lsr_ten.ranks, lsr_ten.separation_rank, lsr_ten.order
    lambda1 = hypers["weight_decay"]
    max_iter = hypers["max_iter"]
    threshold = hypers["threshold"]
    b_intercept = intercept

    #Create models for each factor matrix and core matrix
    factor_matrix_models = [[Ridge(alpha = lambda1, solver = 'svd', fit_intercept = intercept) for k in range(len(ranks))] for s in range(sep_rank)]
    core_tensor_model = Ridge(alpha = lambda1, solver = 'svd', fit_intercept = intercept)

    #Store objective function values
    objective_function_values = np.ones(shape = (max_iter, sep_rank, len(ranks) + 1)) * np.inf
    
    #store gradient norms 
    gradient_values = np.ones(shape = (max_iter, sep_rank, len(ranks) + 1)) * np.inf

    #Normalized Estimation Error
    iterations_normalized_estimation_error = np.zeros(shape = (max_iter,))
    
    #saving iterate level information
    iterate_level_values = [[[[]for _ in range(max_iter) ]for _ in range ((len(ranks)+1))] for _ in range(sep_rank)]
    
    #to save the lsr ten object
    factor_core_iteration = []
    
    #reconstructed tensor
    
    #Run at most max_iter iterations of Block Coordinate Descent
    for iteration in range(max_iter):
        factor_residuals = np.zeros(shape = (sep_rank, len(ranks)))
        core_residual = 0

        #Store updates to factor matrices and core tensor
        updated_factor_matrices = np.empty((sep_rank, len(ranks)), dtype=object)
        updated_core_tensor = None

        #Iterate over the Factor Matrices.
        for s in range(sep_rank):
            for k in range(len(ranks)):
                #Absorb Factor Matrices into X aside from (s, k) to get X_tilde

                X, y = training_data, training_labels
                X_tilde, y_tilde = lsr_ten.bcd_factor_update_x_y(s, k, X, y) #y tilde should now be y-b-<Q,X>
                

                #Solve the sub-problem pertaining to the factor tensor
                factor_matrix_models[s][k].fit(X_tilde, y_tilde)

                #Retrieve Original and Updated Factor Matrices
                Bk = lsr_ten.get_factor_matrix(s, k)
                Bk1 = factor_matrix_models[s][k].coef_
                if intercept: b = factor_matrix_models[s][k].intercept_



                #Shape Bk1 as needed
                Bk1 = np.reshape(Bk1, (shape[k], ranks[k]), order = 'F')
                iterate_level_values[s][k][iteration] = Bk1


                #Update Residuals and store updated factor matrix
                factor_residuals[s][k] = np.linalg.norm(Bk1 - Bk)
                updated_factor_matrices[s, k] = Bk1

                #Update Factor Matrix
                lsr_ten.update_factor_matrix(s, k, updated_factor_matrices[s, k])

                #update the intercept
                if intercept: lsr_ten.update_intercept(b)

                #Calculate Objective Function Value
                expanded_lsr = lsr_ten.expand_to_tensor()
                expanded_lsr = np.reshape(expanded_lsr, X[0].shape, order = 'F')
                objective_function_value = objective_function_tensor_sep(y, X, expanded_lsr,lsr_ten, lambda1, b if intercept else None)# CHECK THISSSS______________________________
                objective_function_values[iteration, s, k] = objective_function_value
                
                #Calculate Gradient Values
                bk = np.reshape(Bk, (-1, 1), order = 'F') #Flatten Factor Matrix Column Wise
                Omega = X_tilde
                z = b if intercept else 0
                gradient_value = (-2 * Omega.T) @ (y_tilde.reshape(-1,1) - Omega @ bk  - z) + (2 * lambda1 * bk)
                
                #Store Gradient Values
                gradient_values[iteration, s, k] = np.linalg.norm(gradient_value, ord = 'fro')

                #Print Objective Function Values
                # print(f"Iteration: {iteration}, Separation Rank: {s}, Factor Matrix: {k}, Objective Function Value: {objective_function_value}")


        #Absorb necessary matrices into X, aside from core tensor, to get X_tilde
        X, y = training_data, training_labels
        X_tilde, y_tilde = lsr_ten.bcd_core_update_x_y(X, y)

        #Solve the sub-problem pertaining to the core tensor
        core_tensor_model.fit(X_tilde, y_tilde)

        #Get Original and Updated Core Tensor
        Gk = lsr_ten.get_core_matrix()
        Gk1 = np.reshape(core_tensor_model.coef_, ranks, order = 'F')
        if intercept: b = core_tensor_model.intercept_
        
        #saving iterate
        iterate_level_values[0][len(ranks)][iteration] = Gk1
        
        #Update Residuals and store updated Core Tensor
        core_residual = np.linalg.norm(Gk1 - Gk)
        updated_core_tensor = Gk1

        #Update Core Tensor
        lsr_ten.update_core_matrix(updated_core_tensor)

        #Update Intercept

        if intercept: lsr_ten.update_intercept(b)

        #Calculate Objective Function Value
        if intercept: b = lsr_ten.get_intercept()
        expanded_lsr = lsr_ten.expand_to_tensor()
        expanded_lsr = np.reshape(expanded_lsr, X[0].shape, order = 'F')
        
        objective_function_value = objective_function_tensor_sep(y, X, expanded_lsr,lsr_ten,lambda1, b if intercept else None)
        objective_function_values[iteration, :, (len(ranks))] = objective_function_value
        
        #Calculate Gradient Values
        g = np.reshape(Gk, (-1, 1), order = 'F') #Flatten Core Matrix Column Wise
        Omega = X_tilde
        z = b if intercept else 0
        gradient_value = (-2 * Omega.T) @ (y_tilde.reshape(-1,1) - Omega @ g  - z) + (2 * lambda1 * g)
        
        #Store Gradient Value
        gradient_values[iteration, :, (len(ranks))] = np.linalg.norm(gradient_value, ord='fro')
        
        #saving lsr_ten
        factor_core_iteration.append(copy.deepcopy(lsr_ten))

        #Print Objective Function Value
        # print(f"BCD Regression, Iteration: {iteration}, Core Tensor, Objective Function Value: {objective_function_value}")

        #Stopping Criteria
        diff = np.sum(factor_residuals.flatten()) + core_residual  #need to change this
        # print('------------------------------------------------------------------------------------------')
        # print(f"Value of Stopping Criteria: {diff}")
        # print(f"Expanded Tensor: {expanded_lsr}")
        # print('------------------------------------------------------------------------------------------')
        if diff < threshold: 
            print('stopping_criterion_reached')
            break
            
    return lsr_ten, objective_function_values, gradient_values, iterate_level_values,factor_core_iteration