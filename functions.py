import numpy as np


##=======================================================================##

def AND(x1, x2):

    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b

    if tmp <= 0:
        y = 0
    else:
        y = 1

    return y

##=======================================================================##

def NAND(x1, x2):

    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5]) ## AND 게이트의 가중치와 편형과 부호만 다르다.
    b = 0.7
    tmp = np.sum(w*x) + b

    if tmp <= 0:
        y = 0
    else:
        y = 1

    return y

##=======================================================================##

def OR(x1, x2):

    x = np.array([x1, x2])
    w = np.array([1.0, 1.0])
    b = -0.9
    tmp = np.sum(w*x) + b

    if tmp <= 0:
        y = 0
    else:
        y = 1

    return y

##=======================================================================##

def XOR(x1, x2):

    s1 = self.NAND(x1, x2)
    s2 = self.OR(x1, x2)
    y = self.AND(s1, s2)

    return y

##=======================================================================##

def step_function(x):

    y = x > 0

    return y.astype(np.int64)

##=======================================================================##

def sigmoid(x):

    y = 1 / (1 + np.exp(-x))

    return y

##=======================================================================##

def ReLU(x):
    w = x > 0
    y = x * w.astype(np.int64)

    return y

##=======================================================================##
"""
def softmax(x):
    c = np.max(x, axis=0)
    exp_x = np.exp(x - c)
    y = exp_x / np.sum(exp_x)

    return y
"""
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

##=======================================================================##

def sum_squares_error(y, t):

    sse = 0.5 * np.sum((y-t)**2)

    return sse

##=======================================================================##

def cross_entropy_error(y, t):

    if y.ndim == 1:    ## 데이터가 하나인 경우
        y = y.reshape(1, y.size)   
        t = t.reshape(1, t.size)

    #if t.size == y.size:
    #    t = t.argmax(axis=1)

    batch_size = y.shape[0]

    return -np.sum(t*np.log(y)) / batch_size


##=======================================================================##

def numerical_diff(f, x):

    h = 1e-4
    dif = (f(x+h) - f(x-h)) / (2*h)

    return dif

##=======================================================================##

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)

        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   

    return grad

##=======================================================================##
"""                         
def numerical_gradient(f, x):

    if x.ndim == 1:
        x = x.reshape(1, x.size)

    h = np.zeros((x.shape), dtype=np.float32)
    grd = np.zeros((x.shape), dtype=np.float32)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            h[i,j] += 1e-4
            print(h)
            grd[i,j] += ((f(x+h) - f(x-h))) / (2*np.sum(h))
            h[i,j] -= 1e-4

    return grd
"""
##=======================================================================##

def gradient_descent(f, init_x, lr=0.01, step_num=100):

    x = init_x

    for i in range(step_num):
        grd = numerical_gradient(f, x)
        x -= lr * grd

    return x

##=======================================================================##