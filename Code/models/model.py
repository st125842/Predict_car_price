import pandas as pd
import numpy as np
import math
from sklearn.model_selection import KFold
import math 
import numpy as np

class LinearRegression(object):
    
    #in this class, we add cross validation as well for some spicy code....
    kfold = KFold(n_splits=3)
            
    def __init__(self, regularization=None, lr=0.001, method='batch',weight_init='zero',use_momentum=True,degree=1,momentum=0.9, num_epochs=500, batch_size=50, cv=kfold):
        self.lr         = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.method     = method
        self.cv         = cv
        self.regularization = regularization
        self.momentum = momentum
        self.prev_step = 0
        self.degree = degree
        self.weight_init = weight_init
        self.use_momentum = use_momentum

    def mse(self, ytrue, ypred):
        # check if scalar
        if np.isscalar(ytrue) or ytrue.shape == ():
            return (ypred - ytrue) ** 2
        else:
            return ((ypred - ytrue) ** 2).sum() / ytrue.shape[0]
    
    def r2(self,ytrue,ypred):
        y_mean = np.mean(ytrue)
        ss_tot = np.sum((ytrue - y_mean) ** 2)
        ss_res = np.sum((ytrue - ypred) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        return r_squared
    
    def xaviai_initialize(self,n_input,n_output):
        lower,upper = -(1.0/np.sqrt(n_input)), (1.0/np.sqrt(n_input))
        numbers = np.random.rand(n_output)
        scaled = lower + numbers*(numbers-lower)
        return scaled
            
    def fit(self, X_train, y_train):
            
        #create a list of kfold scores
        self.kfold_scores = list()
        
        #reset val loss
        self.val_loss_old = np.infty

        #kfold.split in the sklearn.....
        #5 splits
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):
            
            X_cross_train = X_train[train_idx]
            y_cross_train = y_train[train_idx]
            X_cross_val   = X_train[val_idx]
            y_cross_val   = y_train[val_idx]
            if self.degree > 1:
                X_cross_train = self.polynomial_features(X_cross_train, self.degree)
                X_cross_val = self.polynomial_features(X_cross_val, self.degree)

            
            
            if (self.weight_init == 'zero'):
                self.theta = np.zeros(X_cross_train.shape[1])
            else:
                self.theta = self.xaviai_initialize(X_cross_train.shape[1],X_cross_train.shape[1])
            #define X_cross_train as only a subset of the data
            #how big is this subset?  => mini-batch size ==> 50
#             print(self.theta.shape)
#             print()
            #one epoch will exhaust the WHOLE training set
            with mlflow.start_run(run_name=f"Fold-{fold}", nested=True):
                
                params = {"method": self.method, "lr": self.lr, "reg": type(self).__name__}
                mlflow.log_params(params=params)
                for epoch in range(self.num_epochs):
                
                    #with replacement or no replacement
                    #with replacement means just randomize
                    #with no replacement means 0:50, 51:100, 101:150, ......300:323
                    #shuffle your index
                    perm = np.random.permutation(X_cross_train.shape[0])
                            
                    X_cross_train = X_cross_train[perm]
                    y_cross_train = y_cross_train[perm]
                    
                    
                    if self.method == 'sto':
                        for batch_idx in range(X_cross_train.shape[0]):
                            X_method_train = X_cross_train[batch_idx].reshape(1, -1) #(11,) ==> (1, 11) ==> (m, n)
                            y_method_train = y_cross_train[batch_idx] 
#                             print(batch_idx)
#                             print(y_cross_train.reshape(-1).shape)
#                             print(y_method_train)
#                             print(y_method_train.shape)
                            train_loss = self._train(X_method_train, y_method_train)
                    elif self.method == 'mini':
                        for batch_idx in range(0, X_cross_train.shape[0], self.batch_size):
                            #batch_idx = 0, 50, 100, 150
                            X_method_train = X_cross_train[batch_idx:batch_idx+self.batch_size, :]
                            y_method_train = y_cross_train[batch_idx:batch_idx+self.batch_size]
                            train_loss = self._train(X_method_train, y_method_train)
                    else:
                        X_method_train = X_cross_train
                        y_method_train = y_cross_train
                        train_loss = self._train(X_method_train, y_method_train)

                    mlflow.log_metric(key="train_loss", value=train_loss, step=epoch)

                    yhat_val = self.predict(X_cross_val)
                    val_loss_new = self.mse(y_cross_val, yhat_val)
                    val_r2 = self.r2(y_cross_val, yhat_val)
                    mlflow.log_metric(key="val_loss", value=val_loss_new, step=epoch)
                    mlflow.log_metric(key="val_r2", value=val_r2, step=epoch)
                    
                    #early stopping
                    if np.allclose(val_loss_new, self.val_loss_old):
                        break
                    self.val_loss_old = val_loss_new
            
                self.kfold_scores.append(val_loss_new)
                print(f"Fold {fold}: MSE - {val_loss_new}, R2 - {val_r2}")
            
    # def update():

    
    def _train(self, X, y):
        yhat = self.predict(X)
        m    = X.shape[0]        
        reg_term = 0 if self.regularization is None else self.regularization.derivation(self.theta)
        grad = (1/m) * X.T @(yhat - y) + reg_term
        if self.use_momentum:
            step = self.lr*grad
            self.theta = self.theta - step + self.momentum*self.prev_step 
            self.prev_step = step
        else:
            self.theta = self.theta - self.lr * grad
        return self.mse(y, yhat)
    
    def predict(self, X,poly=False):
        if poly:
            X = self.polynomial_features(X,self.degree)
        return X @ self.theta  #===>(m, n) @ (n, )
    
    def _coef(self):
        return self.theta[1:]  #remind that theta is (w0, w1, w2, w3, w4.....wn)
                               #w0 is the bias or the intercept
                               #w1....wn are the weights / coefficients / theta
    def _bias(self):
        return self.theta[0]
    
    def polynomial_features(self,X, degree):
        X_poly = np.ones((X.shape[0], 1))  # column of 1s for bias term
        for d in range(1, degree + 1):
            X_poly = np.hstack((X_poly, X ** d))
        return X_poly
    

class LassoPenalty:
    
    def __init__(self, l):
        self.l = l # lambda value
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.abs(theta))
        
    def derivation(self, theta):
        return self.l * np.sign(theta)
    
class RidgePenalty:
    
    def __init__(self, l):
        self.l = l
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.square(theta))
        
    def derivation(self, theta):
        return self.l * 2 * theta

class Lasso(LinearRegression):
    
    def __init__(self, method, lr, l,weight_init,use_momentum):
        self.regularization = LassoPenalty(l)
        super().__init__(self.regularization, lr, method,weight_init,use_momentum)
        
class Ridge(LinearRegression):
    
    def __init__(self, method, lr, l,weight_init,use_momentum):
        self.regularization = RidgePenalty(l)
        super().__init__(self.regularization, lr, method,weight_init,use_momentum)
        
class Polynomial(LinearRegression):
    def __init__(self, method,lr,weight_init,use_momentum,degree):
#         self.reg = regularization
        super().__init__(None, lr,method,weight_init,use_momentum,degree=degree )
    
                    
