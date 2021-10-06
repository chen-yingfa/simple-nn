from model import Model

class SGD:
    '''
    A (extremely) simple SGD that implement linear decay of learning rate.
    '''
    def __init__(self, model: Model, lr: float, total_steps: int):
        self.model = model
        self.max_lr = lr
        self.total_steps = total_steps

        # Start from max_lr, linearly decay to zero when reaching total_steps
        self.lr = self.max_lr
        self.decay_rate = - lr / total_steps

    def step(self) -> None:
        self.model.update_weights(self.lr)
        self.lr += self.decay_rate  # Learning rate decay

