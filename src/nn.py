import numpy as np

class Component:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Loss:
    '''
    Base class for loss functions
    
    In PyTorch, loss is an object that store the output of loss function,
    and (imo. strangely) has the method backward and is used to trigger a 
    backward pass, which tells all parameters to compute its corresponding
    gradient.
    '''
    pass


class MSELoss(Loss):
    def __call__(self, label: int, preds: np.ndarray) -> float:
        '''
        Return MSE loss. Note that labels are given as integers, and NOT
        an one-hot representation.
        '''
        onehot = np.zeros(preds.size)
        onehot[label] = 1

        # Store for calculating grad
        self.onehot = onehot
        self.preds = preds
        return np.sum((onehot - preds) ** 2) / 2

    def backward(self) -> np.ndarray:
        return self.preds - self.onehot  # grad of loss wrt preds


class CELoss(Loss):
    '''
    Cross-Entropy loss, 
    '''
    def softmax(self, x: np.ndarray) -> np.ndarray:
        '''Numerical stable version of softmax'''
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    def __call__(self, label: int, preds: np.ndarray) -> float:
        '''
        Return the loss as float. Not that labels are given as integers,
        NOT a one-hot representation.
        
        `preds`: 1d array
        '''
        self.label = label
        self.p = self.softmax(preds)
        log_likelihood = -np.log(self.p[label])
        return log_likelihood


    def backward(self) -> np.ndarray:
        self.p[self.label] -= 1.0
        return self.p


class Sigmoid(Component):
    '''
    Implement a sigmoid function, does not hold parameters, but will store
    temporary previous inputs for gradient calculations, so don't reuse it
    multiple times in a model.
    '''
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = 1 / (1 + np.exp(-x))
        return self.x

    def backward(self, grad_next: np.ndarray) -> np.ndarray:
        return grad_next * self.x * (1 - self.x)


class Relu(Component):
    '''
    Implement a ReLU function, does not hold parameters, but will store
    temporary previous inputs for gradient calculations, so don't reuse it
    multiple times in a model.
    '''
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        x[x < 0] = 0
        return x

    def backward(self, grad_next: np.ndarray) -> np.ndarray:
        return grad_next * (self.x >= 0).astype(int)


class Linear(Component):
    '''
    Implements the a Linear transformation (which is a fully connected layer).
    '''
    def __init__(self, len_src: int, len_dst: int):
        '''
        `len_src`: length of the input array
        `len_dst`: length of output array
        '''
        self.W = np.random.randn(len_src, len_dst)
        self.b = np.random.randn(len_dst)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x  # Store for grad calculations
        return x @ self.W + self.b
    
    def backward(self, grad_next: np.ndarray) -> np.ndarray:
        self.grad_b = grad_next
        self.x = np.reshape(self.x, (-1, 1))
        grad_next = grad_next.reshape((1, -1))

        self.grad_W = self.x @ grad_next
        # grad_x is passed to previous layers
        grad_x = grad_next @ self.grad_W.transpose(-1, -2)
        return grad_x

    def update_weights(self, lr: float) -> None:
        '''`lr`: float, learning rate'''
        self.W -= lr * self.grad_W.squeeze()
        self.b -= lr * self.grad_b.squeeze()


if __name__ == '__main__':
    # For debugging
    loss = CELoss()
    y = 1
    x = np.array([-1111, 1])
    print(loss(y, x))
