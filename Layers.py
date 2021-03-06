import numpy as np
from functions  import *

    ##==========================class MulLayer===========================##

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
        
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        
        return out
    
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        
        return dx, dy
    
    ##===========================class AddLayer==========================##

class AddLayer:
    def __init__(self):
        pass
        
    def forward(self, x, y):
        out = x + y
        
        return out
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        
        return dx, dy
    
    ##===========================class ReLULayer==========================##

class ReLULayer:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
            
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx
    
    ##=========================class SigmoidLayer=========================##

class SigmoidLayer:
    def __init__(self):
        self.out = None
                
    def forward(self, x):
        
        out = sigmoid(x)
        
        self.out = out
        
        return out
    
    def backward(self, dout):
        
        dx = dout * self.out * (1.0 - self.out)
        
        return dx

    ##=========================class AffineLayer==========================##

class AffineLayer:
    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        
    def forward(self, x):
        self.x = x
        out = self.x @ self.W + self.b
        
        return out
    
    def backward(self, dout):
        dx = dout @ self.W.T
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0)
        
        return dx
    
    ##=======================class SoftmaxLossLayer=======================##

class SoftmaxLossLayer:
    def __init__(self):
        self.Loss = None ## ??????
        self.t = None    ## ?????? ?????????
        self.out = None  ## softmax??? ??????
                
    def forward(self , x, t):
        self.t = t        
        self.out = softmax(x)
        self.Loss = cross_entropy_error(self.out, self.t)
        
        return self.Loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.out - self.t) / batch_size
        ## batch_size??? ????????? ????????? "/"???????????? ????????? ????????? ????????? ??????. ????????? ???????????? ?????? ????????????
        ## ?????? (?????????) * 1??? ????????? batch?????? ?????? ????????? ?????? ??????????????? (?????????) * batch_size???
        ## ??????. ????????? ????????? ???????????? ????????? ?????? ?????????, ??? dx??? (?????????) / batch_size??? ?????? ?????????.
        
        return dx
    
    ##=========================class DropoutLayer========================##
    
class DropoutLayer:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
        
    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        
        else:
            return x * (1.0 - self.dropout_ratio)
    
    def backward(self, dout):
        return dout * self.mask