### CODE FOR DATA GENERATION

import numpy as np

#Compute Mean and Energy of a given Matrix M
def mean_and_energy(M: np.ndarray):
    mu = np.mean(M)
    s = M.size
    energy = np.sum(np.square(M - mu)) / (s - 1)
    return mu, energy

#n_train: number of training data samples
#n_test: number of test data samples
#tensor_dimensions: The dimensionality of the tensor.
#tensor_mode_ranks: Ranks along each mode of tensor
#separation_rank: S for LSR Decomposition
def generate_data(n_train, n_test, tensor_dimensions: np.ndarray,\
                  tensor_mode_ranks: np.ndarray, separation_rank):

    #Calculate number of tensor dimensions
    D = len(tensor_dimensions)

    #generating the training and testing together
    output_tensor_shape = tuple(np.append(n_train + n_test , tensor_dimensions))

    #Generate Core Tensor
    size = np.prod(tensor_mode_ranks)
    G1 = np.random.normal(0, 1, size)
    G1 = G1.reshape(tensor_mode_ranks)
    G1 = G1/np.linalg.norm(G1,ord = 'fro')


    #generating the factor matrices
    all_factor_matrices=[]
    for s in range (separation_rank):
        mode_s_factormatrices = []
        for k in range(D):
            dummy_mat = np.random.normal(0, 1, size = (tensor_dimensions[k], tensor_mode_ranks[k]) )
            mode_s_factormatrices.append(dummy_mat)
        all_factor_matrices.append(mode_s_factormatrices)

    print('Initiallizng Core Tensor and Factor Matrices: Done')

    #Using the Factor Matrices and Core Tensor, Generate B
    B_tensor_prior_N = np.zeros(tensor_dimensions)
    for s in range(separation_rank):
        B_tensor_prior_N += all_factor_matrices[s][0] @ G1 @ all_factor_matrices[s][1].T

    # Normalizing 
    B_tensor = B_tensor_prior_N
    print('Generating the Parameter Tensor: Done')

    #Generate X
    X = np.random.normal(0,0.5,size=(output_tensor_shape))

    #generating the dependent variable
    B_tensor_flatten = B_tensor.flatten(order ='F')
    X_transpose = np.transpose(X, (0,2,1))
    X_transpose_vectorized = np.reshape(X_transpose, newshape = (X_transpose.shape[0],-1))
    y_no_noise = np.dot(X_transpose_vectorized, B_tensor_flatten)
    y = y_no_noise + np.random.normal(0, 0.01, size = y_no_noise.shape)
    print('Generating Data: Done')

    #Split into X_train, X_test, Y_train, Y_test
    X_train = X[:n_train]
    X_test  = X[n_train:n_train+n_test]
    y_train = y[:n_train]
    y_test  = y[n_train:n_train+n_test]
    print('Splitting into Train/Test: Done')
    
    #Calculating Means and Energies
    factor_matrix_means = np.zeros(shape = (separation_rank, D))
    factor_matrix_energies = np.zeros(shape = (separation_rank, D))
    for s in range(separation_rank):
        for k in range(D):
            factor_matrix_mean, factor_matrix_energy = mean_and_energy(all_factor_matrices[s][k])
            factor_matrix_means[s, k] = factor_matrix_mean
            factor_matrix_energies[s, k] = factor_matrix_energy
    
    core_tensor_mean, core_tensor_energy = mean_and_energy(G1)
    parameter_tensor_mean, parameter_tensor_energy = mean_and_energy(B_tensor)
    X_train_mean, X_train_energy = mean_and_energy(X_train)
    y_train_mean, y_train_energy = mean_and_energy(y_train)
    
    print("Factor Matrix Means: ", factor_matrix_means.flatten(), " Factor Matrix Energies: ", factor_matrix_energies.flatten())
    print("Core Tensor Means: ", core_tensor_mean, " Core Tensor Energies: ", core_tensor_energy)
    print("Parameter Tensor Means: ", parameter_tensor_mean, " Parameter Tensor Energies: ", parameter_tensor_energy)
    print("X_train Means: ", X_train_mean, " X_train Energies: ", X_train_energy)
    print("y_train Means: ", y_train_mean, " y_train Energies: ", y_train_energy)

    return X_train, y_train, X_test, y_test, B_tensor