from typing import Tuple
import torch
from torch import nn
import numpy as np
import copy

def one_hot(arr):
    # determine the number of unique values in the array
    n_values = len(np.unique(arr))
    
    if n_values == 1: # In case all elements were the same, like all 0
        n_values += 1

    # create an empty one-hot encoded array
    one_hot = np.zeros((len(arr), n_values), dtype="float32")

    # one-hot encode the array
    one_hot[np.arange(len(arr)), arr] = 1
    
    return one_hot

class EnhAdaBoostNN:
    def __init__(self, max_num_estimators: int, patience: int, max_epochs: int, nn_topology: Tuple, lr: float, beta: float = 0.5):
        """
        Initializes an instance of the Enhanced AdaBoost algorithm with feedforward neural network as the weak learner.

        Args:
            max_num_estimators (int): The maximum number of weak learners.
            patience (int): The number of iterations to wait before stopping the training process if no improvement is observed.
            max_epochs (int): The maximum number of epochs for training each weak learner.
            nn_topology (list): A tuple of integers representing the topology of the neural network.
            lr (float): The learning rate for the neural network.
            beta (float): The Beta parameter for calculating the weak learner's weight.

        Returns:
            None
        """
        self.alphas = []
        self.estimators = []
        self.training_errors = []

        self.max_num_estimators = max_num_estimators
        self.patience = patience
        self.beta = beta
        self.max_epochs = max_epochs
        self.nn_topology = nn_topology
        self.LR = lr
        self.device = "cpu"
        self.base_estimator = nn.Sequential()
        for l in range(len(self.nn_topology)):
            self.base_estimator.append(nn.Linear(self.nn_topology[l], self.nn_topology[l+1]))
            if l < len(self.nn_topology) - 2:
                self.base_estimator.append(nn.ReLU())
            else:
                break
        self.base_estimator.to(self.device)
        
        
    def __compute_error(self, y, y_pred, D_i):
        return np.sum(D_i * (np.not_equal(y, y_pred)).astype(int))
    
    
    def __compute_sen(self, y, y_pred, D_i):
        P = np.sum(((y == 1) & (y_pred == 1)) * D_i)
        K = np.sum((y == 1) * D_i)
        
        return P / K
    
    
    def __compute_alpha(self, epsilon, gamma, beta, b):
        alpha = None
        restrict = 0.5 * (1 - (2 * gamma - 1)/(b + 1))
        a = (2 * gamma - 1) / (b + 1)
        kappa = (0.5 * np.log((1 + a) / (1 - a))) / np.exp(beta * (2 * gamma - 1))
        
        if (gamma > 0.5) and (epsilon < restrict):
            pos_acc = kappa * np.exp(beta * (2 * gamma - 1))
            alpha = 0.5 * np.log((1 - epsilon) / epsilon) + pos_acc
                    
#         print("res : {:.4f}    e : {:4f}    gamma : {:.4f}    kappa : {:.4f}    "
#               .format(restrict, epsilon, gamma, kappa))        
        
        return alpha
    
    
    def __update_weights(self, D_i, alpha, y, y_pred):
        
        D_i = D_i * np.exp(-alpha * y * y_pred)
        
        return D_i / np.sum(D_i)
    
    
    def __pred_estimator(self, estimator, X):
        estimator.eval()
        with torch.no_grad():
            X = torch.from_numpy(X).to(self.device)
            pred = estimator(X).detach().cpu().numpy()
            pred = np.argmax(pred, axis=1)
            pred = pred * 2 - 1 # (0, 1) -> (-1, 1)
            return pred
    
        
    def __fit_estimator(self, trainX, trainy, D_i):
        trainX, trainy = self.__bootstrap_sample(trainX, trainy, D_i)
        trainy = (trainy + 1) // 2  # (-1, 1) -> (0, 1)
        trainy = one_hot(trainy)
        estimator = copy.deepcopy(self.base_estimator)
        
        optim = torch.optim.Adam(estimator.parameters(), lr=self.LR)    
        criterion = nn.CrossEntropyLoss()
        self.__fit_NN(estimator, optim, criterion, trainX, trainy, self.max_epochs, 8) 
        
        return estimator
    
    
    def __fit_NN_nobatch(self, estimator, optimizer, criterion, X, y, max_epochs):
        estimator.train(True)
        total_iters = 0
        
        X = torch.from_numpy(X).to(self.device)
        y = torch.from_numpy(y).to(self.device)
        
        for i in range(max_epochs):
            y_pred = estimator(X)
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
    def __fit_NN(self, estimator, optimizer, criterion, X, y, max_epochs, batch_size):
        n_batches = X.shape[0] // batch_size
        X_batches = np.array_split(X, n_batches)
        y_batches = np.array_split(y, n_batches)
        
        estimator.train(True)
        total_iters = 0
        
        for i in range(max_epochs):
            total_loss = 0
            
            for X_batch, y_batch in zip(X_batches, y_batches):
                X_batch = torch.from_numpy(X_batch).to(self.device)
                y_batch = torch.from_numpy(y_batch).to(self.device)

                y_pred = estimator(X_batch)
                loss = criterion(y_pred, y_batch)
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_iters += 1
                    
                    
    def __bootstrap_sample(self, X, y, weights):
        indices = [i for i in range(len(X))]
        indices = np.random.choice(indices, size=len(indices), p=weights/weights.sum())
        
        return X[indices,:], y[indices]
        
        
    def fit(self, X, y):
        # Clear before calling
        self.alphas = [] 
        self.training_errors = []
        self.estimators = []
        self.weight_dists = []
        
        b = np.sum(y == -1) / np.sum(y == 1)
        D_i = np.ones(len(y)) / len(y) # At first, weights are all the same and equal to 1 / N
        
        t = 0
        none_alphas = 0
        # Iterate over weak classifiers
        while(t < self.max_num_estimators):   
#             print(f"#############{t}#############")
            
            # (a) Fit weak classifier and predict labels
            estimator = self.__fit_estimator(X, y, D_i)
            y_pred = self.__pred_estimator(estimator, X)

            # (b) Compute error and sensitivity
            epsilon = self.__compute_error(y, y_pred, D_i)
            gamma = self.__compute_sen(y, y_pred, D_i)
            
            # (c) Compute alpha
            alpha = self.__compute_alpha(epsilon, gamma, self.beta, b)
            if alpha is None:
                none_alphas += 1
                if none_alphas == self.patience:
                    print("Loop broken!")
                    break
                else:
                    continue
            
            none_alphas = 0
#             print("alpha: {:.4f}".format(alpha))
                
            self.estimators.append(estimator) 
            self.training_errors.append(epsilon)
            self.alphas.append(alpha)
            
            self.weight_dists.append(D_i)
            D_i = self.__update_weights(D_i, alpha, y, y_pred)
            
            t += 1
                        
        assert len(self.estimators) == len(self.alphas)
        
        return self.weight_dists, self.alphas
        
        
    def predict(self, X):
        weak_preds = np.zeros((len(X), len(self.estimators)))
        
        for m in range(len(self.estimators)):
            y_pred_m = self.__pred_estimator(self.estimators[m], X) # (-1, 1)
            y_pred_m = y_pred_m * self.alphas[m]
            weak_preds[:, m] = y_pred_m
        
        weak_preds = weak_preds.sum(axis=1)
        thresh = 1e-6 # To avoid zero sum
        y_pred = np.sign(weak_preds - thresh)

        return y_pred