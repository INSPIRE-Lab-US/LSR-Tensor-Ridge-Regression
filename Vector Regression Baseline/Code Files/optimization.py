## Import Optimization Toolkits
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from collections import defaultdict

#Cost Function[Least Squares]
#||(XW + b) - Y||_2^2
class LeastSquares(nn.Module):
    def __init__(self, input_dim, uses_bias = False):
        super(LeastSquares, self).__init__() #Initialize class
        self.linear = nn.Linear(input_dim, 1, uses_bias) #input_dim = dimension of each sample in X, output_dim = 1 = number of y values for each sample
                
    #Evaluate the Cost Function given X and y
    def evaluate(self, X, Y, reduction = 'sum'):
        mse_loss = nn.MSELoss(reduction = reduction)
        return mse_loss(self.linear(X), Y)

#Cost Function[Least Squares + L2 Regularization Term]
#||(XW + b) - Y ||_2^2 + lambda * ||w||^2_2
class RidgeRegression(nn.Module):
    def __init__(self, input_dim, lmbd, uses_bias = False):
        super(RidgeRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1, uses_bias) #input_dim = dimension of each sample in X, output_dim = 1 = number of y values for each sample
        self.lmbd = lmbd #Ridge Regression Lambda Value
        
    #Evaluate the Cost Function given X and y
    def evaluate(self, X, Y, reduction = 'sum'):
        mse_loss = nn.MSELoss(reduction = reduction)
        return mse_loss(self.linear(X), Y) + self.l2_regularization()
            
    #Calculate value of lambda * ||w||^2_2
    def l2_regularization(self):
        return self.lmbd * (torch.norm(self.linear.weight) ** 2)
    
#Optimize a Cost Function via Stochastic Gradient Descent
#X: Shape n x d where n is the number of samples and d is the number of features
#Y: Shape n x 1 where n is the number of samples
#cost_function_code: 0 for Normal Least Squares, 1 for Ridge Regression
#hypers: hyperparameters
#optimizer_code: 0 for SGD, 1 for Adagrad, 2 for RMSProp
#p_star: estimated optimal value
#W_true: true weights
def SGD(X: np.ndarray, Y: np.ndarray, cost_function_code = 1, hypers = {}, optimizer_code = 0, p_star = 0, W_true = None):
    hypers = defaultdict(int, hypers) #Convert hypers to defaultdict
    
    #Get necessary hyperparameters
    uses_bias = hypers['bias'] #determine whether the bias term is needed
    lmbd = hypers['lambda'] #Lambda for ridge regression
    lr = hypers['lr'] #learning rate
    epochs = hypers['epochs'] #number of epochs
    batch_size = hypers['batch_size'] #batch size to use for SGD
    
    #Get additional hyperparameters
    momentum = hypers['momentum']
    dampening = hypers['dampening']
    nesterov = hypers['nesterov']
    decay_factor = hypers['decay_factor']
    
    #Initialize Cost Function
    if cost_function_code == 0:
        cost_function = LeastSquares(X.shape[1], uses_bias)
    elif cost_function_code == 1:
        cost_function = RidgeRegression(X.shape[1], lmbd, uses_bias)
    
    #Convert X and Y to pytorch tensors
    X = torch.tensor(X, dtype = torch.float32)
    Y = torch.tensor(Y, dtype = torch.float32)
    
    #If W_true is None, set it to a zero vector
    if not isinstance(W_true, np.ndarray):
        W_true = np.zeros(shape = (X.shape[1], 1))
    
    #Initialize Optimizer
    if optimizer_code == 0:
        optimizer = optim.SGD(cost_function.parameters(), lr = lr, momentum = momentum, dampening = dampening, nesterov = nesterov)
    elif optimizer_code == 1:
        optimizer = optim.Adagrad(cost_function.parameters(), lr = lr)
    elif optimizer_code == 2:
        optimizer = optim.RMSprop(cost_function.parameters(), lr = lr, alpha = decay_factor, momentum = momentum)
    
    #Store batch loss values
    loss_values = []
    
    #Store gap to optimality
    gap_to_optimality = []
    
    #Store Metric Values 
    nee_values = []
    nmse_values = []
    corr_values = []
    R2_values = []

    #Training Loop
    for epoch in range(epochs):
        # Shuffle dataset
        indices = torch.randperm(X.size(0))
        X_shuffled = X[indices]
        y_shuffled = Y[indices]
        
        #Get X and Y sample
        X_sample = X_shuffled[0: batch_size]
        Y_sample = y_shuffled[0: batch_size]
            
        # Compute stochastic loss
        stochastic_loss = cost_function.evaluate(X_sample, Y_sample, 'sum')

        # Zero gradients
        optimizer.zero_grad()

        # Backward pass to compute stochastic gradient
        stochastic_loss.backward()

        # Update parameters
        optimizer.step()
                
        #Print and Store batch loss values
        batch_loss = cost_function.evaluate(X, Y, 'sum')
        loss_values.append(batch_loss.item())
        gap_to_optimality.append(batch_loss.item() - p_star)
        
        #Calculate Metrics
        weights = cost_function.linear.weight.data.numpy().reshape((-1, 1))
        bias = cost_function.linear.bias.item() if uses_bias else 0
        X_numpy = X.numpy()
        Y_predicted = X_numpy @ weights + bias
        Y_numpy = Y.numpy()
        
        nee = ((np.linalg.norm(weights - W_true)) ** 2) /  ((np.linalg.norm(W_true)) ** 2)
        nmse = np.sum(np.square((Y_predicted - Y_numpy))) / np.sum(np.square(Y_numpy))
        correlation = np.corrcoef(Y_predicted.flatten(), Y_numpy.flatten())[0, 1]
        R2_score = r2_score(Y_numpy, Y_predicted)
        
        nee_values.append(nee)
        nmse_values.append(nmse)
        corr_values.append(correlation)
        R2_values.append(R2_score)
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {batch_loss:.4f}, Gap to Optimality: {gap_to_optimality[-1]:.4f}, NEE: {nee}, NMSE: {nmse}, Correlation: {correlation}, R2: {R2_score}')

    weights = cost_function.linear.weight.data.numpy().reshape((-1, 1)) #Return weights as numpy array

    #return weights and bias and loss metrics
    if uses_bias:
        return weights, cost_function.linear.bias.item(), loss_values, gap_to_optimality, nee_values, nmse_values, corr_values, R2_values
    else:
        return weights, 0, loss_values, gap_to_optimality, nee_values, nmse_values, corr_values, R2_values