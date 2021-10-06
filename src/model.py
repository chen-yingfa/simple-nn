import numpy as np
from nn import Linear, Relu, Sigmoid, MSELoss


class Model:
    '''
    Base class for all models.
    '''
    pass


class SimpleNN(Model):
    '''Simple feed forward neural network'''
    def __init__(self):
        self.fc0 = Linear(2, 4)
        self.sigmoid0 = Sigmoid()
        self.fc1 = Linear(4, 4)
        self.sigmoid1 = Sigmoid()
        self.fc2 = Linear(4, 2)
        self.sigmoid2 = Sigmoid()

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.fc0(x)
        x = self.sigmoid0(x)
        x = self.fc1(x)
        x = self.sigmoid1(x)
        x = self.fc2(x)
        x = self.sigmoid2(x)
        return x

    def backward(self, grad_loss: np.ndarray) -> None:
        grad = grad_loss
        grad = self.sigmoid2.backward(grad)
        grad = self.fc2.backward(grad)
        grad = self.sigmoid1.backward(grad)
        grad = self.fc1.backward(grad)
        grad = self.sigmoid0.backward(grad)
        grad = self.fc0.backward(grad)

    def update_weights(self, lr: float) -> None:
        self.fc0.update_weights(lr)
        self.fc1.update_weights(lr)
        self.fc2.update_weights(lr)