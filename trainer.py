from multilayernet import MultiLayerNet
from optimizer import SGD
import numpy as np

class Trainer:
    def __init__(self, learning_rate, weight_decay, x_train, t_train, x_val, t_val, epochs=50):
        self.lr = learning_rate
        self.wd = weight_decay
        self.val_acc_list = []
        self.train_acc_list = []
        self.net = MultiLayerNet(input_size=784, hidden_size=100, output_size=10, weight_init="ReLU", wd_lambda=self.wd)
        self.optimizer = SGD(lr=self.lr)
        self.epochs = epochs
        self.x_train = x_train
        self.t_train = t_train
        self.x_val = x_val
        self.t_val = t_val
        
    def train(self):
        train_size = self.x_train.shape[0]
        batch_size = 100
        
        iter_per_epoch = max(train_size / batch_size, 1)
        iters_num = int(self.epochs * iter_per_epoch)
        
        for i in range(iters_num):
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = self.x_train[batch_mask]
            t_batch = self.t_train[batch_mask]
            
            grads = self.net.gradient(x_batch, t_batch)
            self.optimizer.update(self.net.params, grads)
            
            if i % iter_per_epoch == 0:
                train_acc = self.net.accuracy(self.x_train, self.t_train)
                val_acc = self.net.accuracy(self.x_val, self.t_val)
                self.train_acc_list.append(train_acc)
                self.val_acc_list.append(val_acc)
                
        return self.train_acc_list, self.val_acc_list