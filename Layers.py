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
        self.Loss = None ## 손실
        self.t = None    ## 정답 레이블
        self.out = None  ## softmax의 출력
                
    def forward(self , x, t):
        self.t = t        
        self.out = softmax(x)
        self.Loss = cross_entropy_error(self.out, self.t)
        
        return self.Loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.out - self.t) / batch_size
        ## batch_size로 나누는 이유는 "/"노드에서 갈라진 값들이 모이게 된다. 하나의 데이터에 대해 역전파의
        ## 합은 (역전파) * 1이 되므로 batch들이 한데 모이게 되는 노드에서는 (역전파) * batch_size가
        ## 된다. 따라서 하나의 데이터의 영향을 알고 싶다면, 즉 dx는 (역전파) / batch_size가 되는 것이다.
        
        return dx