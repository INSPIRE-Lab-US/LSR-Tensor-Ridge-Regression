from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

import numpy as np
from LSR_Tensor_2D_v1 import LSR_tensor_dot
from lsr_bcd_regression import lsr_bcd_regression
from optimization import inner_product, R2, objective_function_vectorized

def KFoldCV(X_train: np.ndarray, Y_train: np.ndarray, B_tensored: np.ndarray, alphas, k_folds, hypers, intercept = False):
  kfold = KFold(n_splits=k_folds, shuffle=True)

  #Matrix storing Validation Results
  validation_normalized_estimation_error = np.zeros(shape = (k_folds, len(alphas)))
  validation_nmse_losses = np.zeros(shape = (k_folds, len(alphas)))
  validation_correlations = np.zeros(shape = (k_folds, len(alphas)))
  validation_R2_scores = np.zeros(shape = (k_folds, len(alphas)))

  #Define LSR Tensor Hyperparameters
  ranks = hypers['ranks']
  separation_rank = hypers['separation_rank']
  LSR_tensor_dot_shape = tuple(X_train.shape)[1:]
  need_intercept = intercept

  #Initialize LSR Tensors for each alpha iteration
  lsr_tensors = [ [LSR_tensor_dot(shape = LSR_tensor_dot_shape, ranks = ranks, separation_rank = separation_rank, intercept = need_intercept) for fold_idx in range(k_folds)] for idx in range(len(alphas))]

  #Store objective function values for each fold/alpha
  objective_function_information = np.ones(shape = (k_folds, len(alphas), hypers['max_iter'], separation_rank, len(ranks) + 1))

  #Go thru each fold
  for fold, (train_ids, validation_ids) in enumerate(kfold.split(X_train)):
    X_train_updated, Y_train_updated = X_train[train_ids], Y_train[train_ids]
    X_validation, Y_validation = X_train[validation_ids], Y_train[validation_ids]

    for index1, alpha1 in enumerate(alphas):
      hypers['weight_decay'] = alpha1

      lsr_ten, objective_function_values = lsr_bcd_regression(lsr_tensors[index1][fold], X_train_updated, Y_train_updated, hypers, intercept = need_intercept)
      expanded_lsr = lsr_ten.expand_to_tensor()
      expanded_lsr = np.reshape(expanded_lsr, X_validation[0].shape, order='F')
      Y_validation_predicted = inner_product(np.transpose(X_validation, (0, 2, 1)), expanded_lsr.flatten(order ='F')) + lsr_ten.b

      normalized_estimation_error = ((np.linalg.norm(expanded_lsr - B_tensored)) ** 2) /  ((np.linalg.norm(B_tensored)) ** 2)

      #print(f"Y_validation_predicted: {Y_validation_predicted.flatten()}, Y_validation: {Y_validation.flatten()}")
      validation_nmse_loss = np.sum(np.square((Y_validation_predicted.flatten() - Y_validation.flatten()))) / np.sum(np.square(Y_validation.flatten()))
      correlation = np.corrcoef(Y_validation_predicted.flatten(), Y_validation.flatten())[0, 1]
      R2_value = R2(Y_validation.flatten(), Y_validation_predicted.flatten())

      validation_normalized_estimation_error[fold, index1] = normalized_estimation_error
      validation_nmse_losses[fold, index1] = validation_nmse_loss
      validation_correlations[fold, index1] = correlation
      validation_R2_scores[fold, index1] = R2_value

      #Store Objective Function Information
      objective_function_information[fold, index1] = objective_function_values

      print(f"Fold = {fold}, Alpha = {alpha1}, Normalized Estimation Error: {normalized_estimation_error}, NMSE: {validation_nmse_loss}, Correlation: {correlation}, R^2 Score: {R2_value}")

  #Average out validation results
  average_normalized_estimation_error = np.mean(validation_normalized_estimation_error, axis = 0)
  average_validation_nmse_losses = np.mean(validation_nmse_losses, axis = 0)
  average_validation_correlations = np.mean(validation_correlations, axis = 0)
  average_validation_R2_scores = np.mean(validation_R2_scores, axis = 0)

  #Get alpha value that performs the best
  flattened_avg_validation_nmse_losses = average_validation_nmse_losses.flatten()
  lambda1 = alphas[np.argmin(flattened_avg_validation_nmse_losses)]

  return lambda1, validation_normalized_estimation_error, validation_nmse_losses, validation_correlations, validation_R2_scores, objective_function_information


#Run KFold Cross Validation
def KFoldCV_Vectorized(X_train, Y_train, B_tensored: np.ndarray, alphas, k_folds, intercept = False):
    #Flatten B_tensored
    B_true = B_tensored.flatten()

    #Set up KFold Object
    kfold = KFold(n_splits = k_folds, shuffle = True)

    #need intercept 
    need_intercept = intercept 

    #Set up Matrices to Store Validation Results
    validation_normalized_estimation_errors = np.zeros(shape = (k_folds, len(alphas)))
    validation_nmse_losses = np.zeros(shape = (k_folds, len(alphas)))
    validation_correlations = np.zeros(shape = (k_folds, len(alphas)))
    validation_R2_scores = np.zeros(shape = (k_folds, len(alphas)))
    objective_function_values = np.zeros(shape = (k_folds, len(alphas)))

    #Run through each Fold of KFold CV
    for fold, (train_ids, validation_ids) in enumerate(kfold.split(X_train)):
        X_train_updated, Y_train_updated = X_train[train_ids], Y_train[train_ids] #Keep n - 1 Folds for Training
        X_validation, Y_validation = X_train[validation_ids], Y_train[validation_ids] #Hold one out for validation
        Y_train_updated = Y_train_updated.flatten() #Flatten Y_train_updated just as a safe measure
        Y_validation = Y_validation.flatten() #Flatten Y_validation just as a safe measure

        #Go through each alpha value to train a model on the training folds
        for index1, alpha1 in enumerate(alphas):
            #Fit LRR on X_train_updated, Y_train_updated
            lrr_model = Ridge(alpha = alpha1, solver = 'svd', fit_intercept = need_intercept)
            lrr_model.fit(X_train_updated, Y_train_updated)
            w_flattened = lrr_model.coef_.flatten()

            #Using the Fitted LRR Model, generate Y_validation_predicted
            Y_validation_predicted = lrr_model.predict(X_validation).flatten()

            #Compute NEE, NMSE, Correlation, and R^2 Score
            validation_normalized_estimation_error = ((np.linalg.norm(w_flattened - B_true)) ** 2) /  ((np.linalg.norm(B_true)) ** 2)
            validation_nmse_loss = np.sum(np.square((Y_validation_predicted - Y_validation))) / np.sum(np.square(Y_validation))
            validation_correlation = np.corrcoef(Y_validation_predicted, Y_validation)[0, 1]
            validation_R2_score = lrr_model.score(X_validation, Y_validation)

            #the intercept 
            if need_intercept: 
              b = lrr_model.intercept_ 
              print('Train Intercept:',b)


            #Compute objective function value
            validation_objective_function_value = objective_function_vectorized(Y_train, X_train, w_flattened.reshape((-1, 1)), alpha1, b if need_intercept else None)

            #Store NEE, NMSE, Correlation, and R^2 Score in Matrices
            validation_normalized_estimation_errors[fold, index1] = validation_normalized_estimation_error
            validation_nmse_losses[fold, index1] = validation_nmse_loss
            validation_correlations[fold, index1] = validation_correlation
            validation_R2_scores[fold, index1] = validation_R2_score
            objective_function_values[fold, index1] = validation_objective_function_value

            #Print Results
            print(f"Fold = {fold}, Alpha = {alpha1}, NEE: {validation_normalized_estimation_error}, NMSE: {validation_nmse_loss}, Correlation: {validation_correlation}, R^2 Score: {validation_R2_score},Objective Function Value: {validation_objective_function_value}")

    #Average out validation results
    average_validation_normalized_estimation_errors = np.mean(validation_normalized_estimation_errors, axis = 0)
    average_validation_nmse_losses = np.mean(validation_nmse_losses, axis = 0)
    average_validation_correlations = np.mean(validation_correlations, axis = 0)
    average_validation_R2_scores = np.mean(validation_R2_scores, axis = 0)

    #Get alpha value that performs the best
    flattened_avg_validation_nmse_losses = average_validation_nmse_losses.flatten()
    lambda1 = alphas[np.argmin(flattened_avg_validation_nmse_losses)]

    #Return best Alpha as well as ALL Validation Results
    return lambda1, validation_normalized_estimation_errors, validation_nmse_losses, validation_correlations, validation_R2_scores, objective_function_values

#Run KFold Cross Validation
def KFoldCV_Vectorized_HCP(X_train, Y_train, alphas, k_folds, B_tensored = None, intercept = False):
    #Flatten B_tensored
    
    if B_tensored is not None: 
      B_true = B_tensored.flatten() 

    #Set up KFold Object
    kfold = KFold(n_splits = k_folds, shuffle = True)

    #need intercept 
    need_intercept = intercept 

    #Set up Matrices to Store Validation Results
    if B_tensored is not None: 
       validation_normalized_estimation_errors = np.zeros(shape = (k_folds, len(alphas))) 

    validation_nmse_losses = np.zeros(shape = (k_folds, len(alphas)))
    validation_correlations = np.zeros(shape = (k_folds, len(alphas)))
    validation_R2_scores = np.zeros(shape = (k_folds, len(alphas)))
    objective_function_values = np.zeros(shape = (k_folds, len(alphas)))

    #Run through each Fold of KFold CV
    for fold, (train_ids, validation_ids) in enumerate(kfold.split(X_train)):
        X_train_updated, Y_train_updated = X_train[train_ids], Y_train[train_ids] #Keep n - 1 Folds for Training
        X_validation, Y_validation = X_train[validation_ids], Y_train[validation_ids] #Hold one out for validation
        Y_train_updated = Y_train_updated.flatten() #Flatten Y_train_updated just as a safe measure
        Y_validation = Y_validation.flatten() #Flatten Y_validation just as a safe measure

        #Go through each alpha value to train a model on the training folds
        for index1, alpha1 in enumerate(alphas):
            #Fit LRR on X_train_updated, Y_train_updated
            lrr_model = Ridge(alpha = alpha1, solver = 'svd', fit_intercept = need_intercept)
            lrr_model.fit(X_train_updated, Y_train_updated)
            w_flattened = lrr_model.coef_.flatten()

            #Using the Fitted LRR Model, generate Y_validation_predicted
            Y_validation_predicted = lrr_model.predict(X_validation).flatten()

            #Compute NEE, NMSE, Correlation, and R^2 Score
            if B_tensored is not None:
              validation_normalized_estimation_error = ((np.linalg.norm(w_flattened - B_true)) ** 2) /  ((np.linalg.norm(B_true)) ** 2) 
            validation_nmse_loss = np.sum(np.square((Y_validation_predicted - Y_validation))) / np.sum(np.square(Y_validation))
            validation_correlation = np.corrcoef(Y_validation_predicted, Y_validation)[0, 1]
            validation_R2_score = lrr_model.score(X_validation, Y_validation)

            #the intercept 
            if need_intercept: 
              b = lrr_model.intercept_ 
              print('Train Intercept:',b)


            #Compute objective function value
            validation_objective_function_value = objective_function_vectorized(Y_train, X_train, w_flattened.reshape((-1, 1)), alpha1, b if need_intercept else None)

            #Store NEE, NMSE, Correlation, and R^2 Score in Matrices
            if B_tensored is not None:
              validation_normalized_estimation_errors[fold, index1] = validation_normalized_estimation_error 
            validation_nmse_losses[fold, index1] = validation_nmse_loss
            validation_correlations[fold, index1] = validation_correlation
            validation_R2_scores[fold, index1] = validation_R2_score
            objective_function_values[fold, index1] = validation_objective_function_value

            #Print Results
            if B_tensored is not None:
              print(f"Fold = {fold}, Alpha = {alpha1}, NEE: {validation_normalized_estimation_error}, NMSE: {validation_nmse_loss}, Correlation: {validation_correlation}, R^2 Score: {validation_R2_score},Objective Function Value: {validation_objective_function_value}")
            else:
              print(f"Fold = {fold}, Alpha = {alpha1}, NMSE: {validation_nmse_loss}, Correlation: {validation_correlation}, R^2 Score: {validation_R2_score},Objective Function Value: {validation_objective_function_value}")
    
    #Average out validation results
    if  B_tensored is not None:
      average_validation_normalized_estimation_errors = np.mean(validation_normalized_estimation_errors, axis = 0) 
    
    average_validation_nmse_losses = np.mean(validation_nmse_losses, axis = 0)
    average_validation_correlations = np.mean(validation_correlations, axis = 0)
    average_validation_R2_scores = np.mean(validation_R2_scores, axis = 0)

    #Get alpha value that performs the best
    flattened_avg_validation_nmse_losses = average_validation_nmse_losses.flatten()
    lambda1 = alphas[np.argmin(flattened_avg_validation_nmse_losses)]

    #Return best Alpha as well as ALL Validation Results
    if B_tensored is not None:
      return lambda1, validation_normalized_estimation_errors, validation_nmse_losses, validation_correlations, validation_R2_scores, objective_function_values
    else: 
      validation_normalized_estimation_errors = np.zeros(shape = (k_folds, len(alphas))) 
      return lambda1, validation_normalized_estimation_errors, validation_nmse_losses, validation_correlations, validation_R2_scores, objective_function_values