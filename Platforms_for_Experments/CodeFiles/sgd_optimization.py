## Import Optimization Toolkits
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

#Cost Function[Least Squares]
#||(XW + b) - Y||_2^2
class LeastSquares(nn.Module):
    def __init__(self, input_dim, uses_bias = False):
        super(LeastSquares, self).__init__() #Initialize class
        self.linear = nn.Linear(input_dim, 1, uses_bias) #input_dim = dimension of each sample in X, output_dim = 1 = number of y values for each sample
                
    #Evaluate the Cost Function given X and y
    def evaluate(self, X, Y, reduction = 'sum'):
        mse_loss = nn.MSELoss(reduction = reduction)
        return mse_loss(self.linear(X), Y.rehshape(-1,1))

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
    
#Perform Exact Line Search for Ridge Regression
#Ridge Regression: ||(XW + b) - Y ||_2^2 + lambda * ||w||^2_2
def exact_line_search_RR(X: np.ndarray, Y: np.ndarray, lmbd, cost_function, uses_bias):
    #Get Model Parameters
    W = cost_function.linear.weight.data.numpy().reshape((-1, 1)) 
    b = cost_function.linear.bias.item() if uses_bias else 0
    
    #Search Direction
    DeltaW = -1 * cost_function.linear.weight.grad.numpy().reshape((-1, 1))
    Deltab = -1 * cost_function.linear.bias.grad if uses_bias else 0
    
    #Compute value of t
    numerator = -((X@W + b - Y).T @ (X @ DeltaW + Deltab)) - (lmbd * (W.T @ DeltaW))
    denominator = (np.linalg.norm(X @ DeltaW + Deltab) ** 2) + (lmbd * (np.linalg.norm(DeltaW) ** 2))
    t = (numerator / denominator) [0, 0]
    
    return t    
    
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
    
    #Splitting data into minibatches 
    dataset = TensorDataset(X,Y)
    dataloader = DataLoader(dataset, batch_size = batch_size,  shuffle=True )
    
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

    #saving gradient 
    sub_problem_gradient =  []
    
    #Store Metric Values

    nmse_values = []
    corr_values = []
    R2_values = []

    #Training Loop
    for epoch in range(epochs):
    
        num_batches = 0
        sum_gradient_norm = 0

        for X_sample, Y_sample in dataloader:
        
            Y_sample = Y_sample.reshape(-1,1)

            # Compute stochastic loss
            stochastic_loss = cost_function.evaluate(X_sample, Y_sample, 'sum')
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Backward pass to compute stochastic gradient
            stochastic_loss.backward()
            
            # Update parameters
            optimizer.step()
            
            #storing the batch wise gradients 
            for param in cost_function.parameters():
               #print(f"Gradient Norm: {torch.norm(param.grad)}")
               stochastic_gradient  = torch.norm(param.grad)
               sum_gradient_norm += stochastic_gradient
            
            num_batches += 1

        #Print and Store batch loss values
        batch_loss = cost_function.evaluate(X, Y, 'sum')
        loss_values.append(batch_loss.item())
        gap_to_optimality.append(batch_loss.item() - p_star)

        #Zero gradients
        #optimizer.zero_grad()

        # Backward pass to compute stochastic gradient
        #batch_loss.backward()

        #for param in cost_function.parameters():
        #       #print(f"Gradient Norm: {torch.norm(param.grad)}")
        #       gradient_after_epoch = torch.norm(param.grad)
        #       sub_problem_gradient.append(gradient_after_epoch)

        epcoh_gradient = sum_gradient_norm/num_batches
        sub_problem_gradient.append(epcoh_gradient)
        
        #Calculate Metrics
        weights = cost_function.linear.weight.data.numpy().reshape((-1, 1))
        bias = cost_function.linear.bias.item() if uses_bias else 0
        X_numpy = X.numpy()
        Y_predicted = X_numpy @ weights + bias
        Y_numpy = Y.numpy()
        

        nmse = np.sum(np.square((Y_predicted - Y_numpy))) / np.sum(np.square(Y_numpy))
        correlation = np.corrcoef(Y_predicted.flatten(), Y_numpy.flatten())[0, 1]
        R2_score = r2_score(Y_numpy, Y_predicted)
        
        nmse_values.append(nmse)
        corr_values.append(correlation)
        R2_values.append(R2_score)
        
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {batch_loss:.4f}, Gap to Optimality: {gap_to_optimality[-1]:.4f}, NMSE: {nmse}, Correlation: {correlation}, R2: {R2_score}')

    # Create a figure with two horizontal subplots
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plotting in the first subplot (ax1)
    #ax1.plot(range(1, len(loss_values) + 1), loss_values, marker='o')
    #ax1.set_title('Subproblem Loss')
    #ax1.set_yscale('log')
    #ax1.set_xlabel('Epoch')
    #ax1.set_ylabel('Loss')
    #ax1.grid(True)

    # Plotting in the second subplot (ax2)
    #ax2.plot(range(1, len(sub_problem_gradient) + 1), sub_problem_gradient, marker='o')
    #ax2.set_title('Subproblem Gradient')
    #ax2.set_yscale('log')
    #ax2.set_xlabel('Epoch')
    #ax2.set_ylabel('Loss')
    #ax2.grid(True)

    # Adjust the layout to prevent overlap
    #plt.tight_layout()

    # Display the plot
    #plt.show()

    weights = cost_function.linear.weight.data.numpy().reshape((-1, 1)) #Return weights as numpy array

    #return weights and bias and loss metrics
    if uses_bias:
        return weights, cost_function.linear.bias.item(), loss_values, gap_to_optimality, nmse_values, corr_values, R2_values,sub_problem_gradient,loss_values
    else:
        return weights, 0, loss_values, gap_to_optimality, nmse_values, corr_values, R2_values,sub_problem_gradient,loss_values
    

#Optimize a Cost Function via Gradient Descent with Exact Line Search
#X: Shape n x d where n is the number of samples and d is the number of features
#Y: Shape n x 1 where n is the number of samples
#cost_function_code: 0 for Normal Least Squares, 1 for Ridge Regression
#hypers: hyperparameters
#p_star: estimated optimal value
#W_true: true weights
def GD2(X: np.ndarray, Y: np.ndarray, cost_function_code = 1, hypers = {}, p_star = 0, W_true = None):
    hypers = defaultdict(int, hypers) #Convert hypers to defaultdict
    
    #Get necessary hyperparameters
    uses_bias = hypers['bias'] #determine whether the bias term is needed
    lmbd = hypers['lambda'] #Lambda for ridge regression
    epochs = hypers['epochs'] #number of epochs
    
    #Initialize Cost Function
    if cost_function_code == 0:
        cost_function = LeastSquares(X.shape[1], uses_bias)
    elif cost_function_code == 1:
        cost_function = RidgeRegression(X.shape[1], lmbd, uses_bias)
    
    #Convert X and Y to pytorch tensors
    X_tensor = torch.tensor(X, dtype = torch.float32)
    Y_tensor = torch.tensor(Y, dtype = torch.float32)
    
    #If W_true is None, set it to a zero vector
    #if not isinstance(W_true, np.ndarray):
    #    W_true = np.zeros(shape = (X.shape[1], 1))
        
    #Store batch loss values
    loss_values = []
    
    #Store gap to optimality
    gap_to_optimality = []
    
    #Store Metric Values 
    #nee_values = []
    nmse_values = []
    corr_values = []
    R2_values = []

    #Training Loop
    for epoch in range(epochs):
        # Zero the gradients
        for param in cost_function.parameters():
            if param.grad is not None:
                param.grad.zero_()
            
        # Compute loss
        loss = cost_function.evaluate(X_tensor, Y_tensor, 'sum')

        # Backward pass to compute gradient
        loss.backward()

        # Update parameters
        t = exact_line_search_RR(X, Y, lmbd, cost_function, uses_bias)
        print(f"Value of t is: {t}")
        
        # Manually update the weights and biases
        with torch.no_grad():
            for param in cost_function.parameters():
                param -= t * param.grad
        
        #Print and Store loss values
        loss_value = cost_function.evaluate(X_tensor, Y_tensor, 'sum').item()
        loss_values.append(loss_value)
        gap_to_optimality.append(loss_value - p_star)
        
        #Calculate Metrics
        weights = cost_function.linear.weight.data.numpy().reshape((-1, 1))
        bias = cost_function.linear.bias.item() if uses_bias else 0
        X_numpy = X_tensor.numpy()
        Y_predicted = X_numpy @ weights + bias
        Y_numpy = Y_tensor.numpy()
        
        #nee = ((np.linalg.norm(weights - W_true)) ** 2) /  ((np.linalg.norm(W_true)) ** 2)
        nmse = np.sum(np.square((Y_predicted - Y_numpy))) / np.sum(np.square(Y_numpy))
        correlation = np.corrcoef(Y_predicted.flatten(), Y_numpy.flatten())[0, 1]
        R2_score = r2_score(Y_numpy, Y_predicted)
        
        #nee_values.append(nee)
        nmse_values.append(nmse)
        corr_values.append(correlation)
        R2_values.append(R2_score)
                
        #print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss_value:.4f}, Gap to Optimality: {gap_to_optimality[-1]:.4f}, NMSE: {nmse}, Correlation: {correlation}, R2: {R2_score}')
        
        # Stopping Criteria
        criteria_satisfied = True
        for name, param in cost_function.named_parameters():
            if param.grad is not None:
                print(f"Gradient Norm for {name}: {torch.norm(param.grad)}")
                criteria_satisfied = criteria_satisfied and (torch.norm(param.grad) <= 0.001)
            else:
                print(f"No gradient Norm for {name}")
        
        if criteria_satisfied:
            break

    weights = cost_function.linear.weight.data.numpy().reshape((-1, 1)) #Return weights as numpy array

    #return weights and bias and loss metrics
    if uses_bias:
        return weights, cost_function.linear.bias.item(), loss_values, gap_to_optimality, nmse_values, corr_values, R2_values
    else:
        return weights, 0, loss_values, gap_to_optimality, nmse_values, corr_values, R2_values
