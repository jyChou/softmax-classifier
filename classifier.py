import numpy as np
from util import plot_boundary, plot_loss_and_accuracy, load_data
class SGD():
    
    __slots__ = ('momentum', 'learning_rate', 'l2_decay_rate', 'batch_size', 'W', 'epochs', 'loss', 'acc', 'info')
    
    def __init__(self, momentum = 0.05, learning_rate = 0.7, l2_decay_rate = 0.001, batch_size = 10, epochs=1000, info=True):
        
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.l2_decay_rate = l2_decay_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.W = None
        self.loss = {'train':np.zeros(epochs), 'test':np.zeros(epochs)}
        self.acc = {'train':np.zeros(epochs), 'test':np.zeros(epochs)}
        self.info = info

    def train(self, train_x, train_y, test_x, test_y):
        
        categories = np.unique(train_y).shape[0]
        
        # extend one dimension as bias for convenience
        train_x = self.normalize(train_x)
        train_x = np.concatenate((np.ones((train_x.shape[0], 1 )), train_x) , axis = 1).astype(float)
        W = np.random.rand(categories, train_x.shape[1] )
        V = np.zeros(shape = (categories, train_x.shape[1]))
        train_y = self.one_hot(train_y, categories)
        
        test_x = self.normalize(test_x)
        test_x = np.concatenate((np.ones((test_x.shape[0], 1 )), test_x) , axis = 1).astype(float)
        test_y = self.one_hot(test_y, categories)
        for epoch in range(self.epochs):
            
            #shuffle training data
            indices = np.arange(train_x.shape[0])
            np.random.shuffle(indices)
            train_x, train_y = train_x[indices], train_y[indices]
            
            loss_sum = 0
            for batch_X, batch_Y in self.batch(train_x, train_y, self.batch_size):
                grad = self.gradient(batch_X, batch_Y, W)
                V = self.momentum * V  + self.learning_rate * grad
                W =  W  - V
                
            # update weight, loss, and accuracy
            self.W = W
            self.loss['train'][epoch] = self.cross_entropy(train_x, train_y, W) / train_x.shape[0] + 0.5 * self.l2_decay_rate * np.sum (W * W)
            self.loss['test'][epoch] = self.cross_entropy(test_x, test_y, W) / test_x.shape[0] + 0.5 * self.l2_decay_rate * np.sum (W * W)
            self.acc['train'][epoch] = self.mean_accuracy(train_x, train_y)
            self.acc['test'][epoch] = self.mean_accuracy(test_x, test_y)
            
            if self.info and epoch % 100 == 0:
                print("Epoch: {}".format(epoch))
                print("Training Loss: {:.5f}, Testing Loss: {:.5f}".format(self.loss['train'][epoch], self.loss['test'][epoch]))
                print("Training mpc accuracy: {:10.5f}, Testing mpc accuracy: {:10.5f}".format(self.acc['train'][epoch], self.acc['test'][epoch]))
                print('{:*^100}'.format(''))
                
    def batch(self, X, Y, batch_size):
        
        length = X.shape[0]
        for idx in range(0, length, batch_size):
            yield X[idx:min(idx + batch_size, length)], Y[idx:min(idx + batch_size, length)]
    
    def cross_entropy(self, X, Y, W):
        S = self.softmax(X @ W.T) 
        loss = - np.sum( Y * np.log(np.max(S))) 
        return loss
    
    def gradient(self, X, Y, W):
        # substract max to reduce computational overhead
        S = self.softmax(X @ W.T)
        return  (1 / X.shape[0]) * (S - Y).T @ X + self.l2_decay_rate * W
    
    def mean_accuracy(self, X, Y):
        
        W = self.W
        S = self.softmax(X @ self.W.T)
        y_predict = np.argmax(S, axis=1) + 1
        ground_truth = np.argmax(Y, axis=1) + 1
        #print(y_predict.shape, ground_truth.shape)
        return np.sum ( np.equal ( y_predict, ground_truth) ) / ground_truth.shape[0] * 100
    
    def predict(self, X):
        X = self.normalize(X)
        X = np.concatenate((np.ones((X.shape[0], 1 )), X) , axis = 1).astype(float)
        W = self.W
        S = self.softmax(X @ self.W.T)
        y_predict = np.argmax(S, axis=1) + 1
        return y_predict
    
    def one_hot(self, Y, categories):
        ret = np.zeros(shape=(Y.shape[0], categories))
        for idx, value in enumerate(Y):
            ret[idx, value-1] = 1
        return ret
    
    def normalize(self, X):
        return 2 * (X - 0.5)
        
    def softmax(self, a):
        a -= a.max(axis=1, keepdims=True)
        return np.exp(a) / np.sum( np.exp(a), axis=1, keepdims=True)


if __name__ == "__main__":
    
    # load data
    train_data_path = './resources/iris/iris-train.txt'
    test_data_path = './resources/iris/iris-test.txt'
    train_x, train_y = load_data(train_data_path)
    test_x, test_y = load_data(test_data_path)
    
    # momentum, learning rate, l2 decay rate
    params = [0.005, 0.1, 0.01] 
    clr = SGD(*params)
    clr.train(train_x, train_y, test_x, test_y)
    
    # visualize loss and boundary
    plot_loss_and_accuracy(clr)
    plot_boundary(clr, train_x, train_y)