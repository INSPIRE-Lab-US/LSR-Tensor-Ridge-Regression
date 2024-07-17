import numpy as np

# Low Separation Rank Tensor Decomposition
class LSR_tensor_dot():
    # Constructor
    def __init__(self, shape, ranks, separation_rank, dtype = np.float32, intercept = False ,initialize = True):
        super(LSR_tensor_dot, self).__init__()
        self.shape = shape
        self.ranks = ranks
        self.separation_rank = separation_rank
        self.dtype = dtype
        self.order = len(shape)
        self.init_params(initialize)
        self.init_params(intercept)

    # Initialize Parameters
    def init_params(self, intercept = False ,initialize = True):
        # Initialize core tensor as independent standard gaussians
        if not initialize:
            self.core_tensor = np.zeros(shape = self.ranks)
        else:
            self.core_tensor = np.random.normal(size = self.ranks)

        # Set up Factor Matrices
        self.factor_matrices = []

        # Initialize all factor matrices
        for s in range(self.separation_rank):
            factors_s = []
            for k in range(self.order):
                if not initialize:
                    factor_matrix_B = np.eye(self.shape[k])[:, self.ranks[k]]
                    factors_s.append(factor_matrix_B)
                else:
                    factor_matrix_A = np.random.normal(0,1,size= (self.shape[k], self.ranks[k]))
                    factors_s.append(factor_matrix_A)

            self.factor_matrices.append(factors_s)

        if intercept:
          ('intercept is initialized')
          self.b = np. random.normal(0,1)
        else:
          (print('intercept is not initialized'))
          self.b = 0

    # Expand core tensor and factor matrices to full tensor, optionally excluding
    # a given term from the separation rank decomposition
    def expand_to_tensor(self, skip_term = None):
        full_lsr_tensor = np.zeros(shape = self.shape)

        #Calculate Expanded Tensor
        for s, term_s_factors in enumerate(self.factor_matrices):
            if s == skip_term: continue
            expanded_tucker_term = term_s_factors[0] @ self.core_tensor @ term_s_factors[1].T
            full_lsr_tensor += expanded_tucker_term

        #Column Wise Flatten full_lsr_tensor
        full_lsr_tensor = full_lsr_tensor.flatten(order = 'F')
        return full_lsr_tensor

    # Absorb all factor matrices and core tensor into the input tensor except for matrix s, k
    # Used during a factor matrix update step of block coordiante descent
    def bcd_factor_update_x_y(self, s, k, x, y):
        #Take x and swap axes 1 and 2 so that vectorization occurs "COLUMN WISE"
        x_transpose = np.transpose(x, (0, 2, 1))
        x_transpose_vectorized = np.reshape(x_transpose, newshape = (x_transpose.shape[0], -1))

        #if we are unfolding along mode 1, use x. Else, if we are unfolding along mode, use x_transpose
        x_partial_unfold = x if k == 0 else x_transpose

        #If k = 0(skip first factor matrix), we have 2nd factor matrix. If k= 1(skip second factor matrix), we have first factor matrix
        kronecker_term = self.factor_matrices[s][1] if k == 0 else self.factor_matrices[s][0]

        #if k = 0, G^T. Else if k = 1, put G
        core_tensor_term = self.core_tensor.T if k == 0 else self.core_tensor

        omega = x_partial_unfold @ kronecker_term @ core_tensor_term
        omega = np.transpose(omega, (0, 2, 1))
        omega = np.reshape(omega, newshape = (omega.shape[0], -1))

        X_tilde = omega
        y_tilde = y

        if self.separation_rank == 1:
            pass
        else:
            gamma = np.dot(x_transpose_vectorized,self.expand_to_tensor(skip_term = s))
            #gamma = gamma.reshape(-1,1)
            y_tilde = y - gamma

        return X_tilde, y_tilde

    # Absorb all factor matrices the input tensor (not the core tensor)
    # Used during a core tensor update step of block coordiante descent
    def bcd_core_update_x_y(self, x, y):
        #Take x and swap axes 1 and 2 so that vectorization occurs "COLUMN WISE"
        x_transpose = np.transpose(x, (0, 2, 1))
        x_transpose_vectorized = np.reshape(x_transpose, newshape = (x_transpose.shape[0], -1))

        #Calculate y_tilde
        y_tilde = y

        #Calculate Kronecker Factor Sum
        kron_factor_sum = 0
        for term_s_factors in self.factor_matrices:
            kron_factor_sum += np.kron(term_s_factors[1], term_s_factors[0])

        #Return Core Update
        return (kron_factor_sum.T @ x_transpose_vectorized.T).T, y_tilde


    #Retrieve factor matrix
    def get_factor_matrix(self, s, k):
      return self.factor_matrices[s][k]

    #Update factor matrix
    def update_factor_matrix(self, s, k, updated_factor_matrix: np.ndarray):
      self.factor_matrices[s][k] = updated_factor_matrix

    def update_intercept(self,updated_b):
      self.b = updated_b

    #Retrieve Core Matrix
    def get_core_matrix(self):
      return self.core_tensor

    #Update core matrix
    def update_core_matrix(self, updated_core_matrix: np.ndarray):
      self.core_tensor = updated_core_matrix

    #Retrive intercept
    def get_intercept(self):
      return self.b