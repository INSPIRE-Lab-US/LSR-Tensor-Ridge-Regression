# This contain all the helper function to impose the adhoc symmetricity constraint on the factor matrices and the core tensor 

def  impose_symmetry(lsr_tensor):
    
    #making the factor matrices equal for one separation rank
    
    #iterating over the factor matrices 
    for sep in range (len(lsr_tensor.factor_matrices)):
          updated_factor_matrix = (lsr_tensor.factor_matrices[sep][0] + lsr_tensor.factor_matrices[sep][1] )/ 2
          lsr_tensor.update_factor_matrix(sep,0,updated_factor_matrix)
          lsr_tensor.update_factor_matrix(sep,1,updated_factor_matrix)
          
    #projecting the core tensor in to the space of symmetric matrices
    updated_core_tensor = (lsr_tensor.core_tensor + lsr_tensor.core_tensor.T) / 2    
    lsr_tensor.update_core_matrix(updated_core_tensor)
    return 
                  