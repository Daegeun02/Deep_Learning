import numpy as np
from functions import *
from Layers import *
from collections import OrderedDict

class MultiLayerNet:
    
    def __init__(self, input_size, hidden_size, output_size, weight_init="ReLU", wd_lambda=0):
        
        ## 파리미터 초기화
        self.wd_lambda = wd_lambda
        self.neural_size = np.hstack((input_size, hidden_size, output_size))
        self.N = len(self.neural_size)-2
        self.params = {}
        self.init_weight(weight_init)
        
        ## 계층 생성
        self.layers = OrderedDict()
        for n in range(self.N+1):
            self.layers[f"Affine{n+1}"] = AffineLayer(self.params[f"W{n+1}"], self.params[f"b{n+1}"])
            if n != self.N:
                self.layers[f"ReLU{n+1}"] = ReLULayer()
            else:
                self.lastlayer = SoftmaxLossLayer()
                
    def init_weight(self, weight_init):
        
        ## Xavier
        if weight_init == "Xavier":
            for n in range(self.N+1):
                sc = np.sqrt(1/self.neural_size[n])
                self.params[f"W{n+1}"] = sc*np.random.randn(self.neural_size[n], self.neural_size[n+1])
                self.params[f"b{n+1}"] = np.zeros(self.neural_size[n+1])
        
        ## He
        elif weight_init == "ReLU":
            for n in range(self.N+1):
                sc = np.sqrt(2/self.neural_size[n])
                self.params[f"W{n+1}"] = sc*np.random.randn(self.neural_size[n], self.neural_size[n+1])
                self.params[f"b{n+1}"] = np.zeros(self.neural_size[n+1])
        
        else:
            sc = weight_init
            for n in range(self.N+1):
                self.params[f"W{n+1}"] = sc*np.random.randn(self.neural_size[n], self.neural_size[n+1])
                self.params[f"b{n+1}"] = np.zeros(self.neural_size[n+1])
                
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        
        return self.lastlayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
            
        acc = np.sum(y == t) / float(x.shape[0])
        
        return acc
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W:self.loss(x, t)
        
        grads = {}

        for key in self.params.keys():
            grad[key] = numerical_gradient(loss_W, self.params[key])
            
        return grads
    
    def gradient(self, x, t):
        
        ## 순전파
        self.loss(x, t)
        
        ## 역전파
        dout = 1
        dout = self.lastlayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            
        grads = {}
        
        for n in range(self.N+1):
            grads[f"W{n+1}"] = self.layers[f"Affine{n+1}"].dW + \
            self.wd_lambda * self.layers[f"Affine{n+1}"].W
            grads[f"b{n+1}"] = self.layers[f"Affine{n+1}"].db
        
        return grads