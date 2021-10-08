import pickle
import random

import numpy as np

from data_loader import load_examples
from visualize import scatter_examples
from model import SimpleNN, Model
from nn import CELoss, Loss
from optimizer import SGD


def get_acc(labels: list, preds: list) -> float:
    '''
    Return accuracy of predictions given labels
    `labels` and `preds` are list of ints, where each int is the index of
    corresponding class.
    '''
    assert len(labels) == len(preds)
    cnt = 0
    for label, pred in zip(labels, preds):
        cnt += label == pred
    return cnt / len(labels)


def evaluate(model: Model, examples: list, criterion: Loss) -> dict:
    '''
    Return acc and loss of model on given examples.
    '''
    labels = []
    preds = []
    total_loss = 0
    for step, (x, y) in enumerate(examples):
        logits = model.forward(x)
        loss = criterion(y, logits)
        total_loss += loss
        
        pred = np.argmax(logits)  # The index with greated probability (logit)
        preds.append(pred)
        labels.append(y)
    acc = get_acc(labels, preds)
    loss = total_loss / len(examples)
    return {'loss': loss,
            'acc': acc}
    

def train(model, train_examples, dev_examples):
    '''Train a model, and save the best one'''
    epochs = 10
    learn_rate = 1  # Greatest learning rate during training

    total_steps = len(train_examples) * epochs
    optimizer = SGD(model, learn_rate, total_steps)  # Linear decay
    criterion = CELoss()  # Loss function


    random.shuffle(train_examples)
    print('*** Start training ***')
    print(f'# epochs: {epochs}')
    print(f'# training examples: {len(train_examples)}')

    train_losses = []
    dev_losses = []

    for ep in range(epochs):
        total_loss = 0
        for step, (x, y) in enumerate(train_examples):
            # Forward pass
            logits = model.forward(x)
            loss = criterion(y, logits)

            # Backward pass (compute gradients)
            grad_loss = criterion.backward()
            model.backward(grad_loss)

            optimizer.step() # Update weights

            total_loss += loss
            # Log
            if (step + 1) % 1000 == 0:
                cur_avg_loss = total_loss / (step + 1)
                print('train_loss:', cur_avg_loss)
        
        train_loss = total_loss / len(train_examples)
        
        # Evaluation
        dev_result = evaluate(model, dev_examples, criterion)
        dev_acc = dev_result['acc']
        dev_loss = dev_result['loss']
        dev_losses.append(dev_loss)
        print(f'dev_acc = {dev_acc}, dev_loss = {dev_loss}, train_loss = {train_loss}')

        # Save best model
        if len(dev_losses) == 0 or dev_loss == min(dev_losses):
            with open('best_model.pkl', 'wb') as f:
                pickle.dump(model, f)
    print('*** End training ***')


def test(model, examples):
    '''Test best model'''
    criterion = CELoss()
    result = evaluate(model, examples, criterion)
    print('Test result:')
    print('acc:', result['acc'])
    print('loss:', result['loss'])

    # Visualize the model as a transformation of 2D plane
    transformed = []
    for (x, y) in examples:
        logits = model.forward(x)
        transformed.append((logits, y))
    scatter_examples(transformed, xlim=(0, 1), ylim=(0, 1), add_line=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def main():
    set_seed(0)
    # Init a new model
    model = SimpleNN()
    train_examples = load_examples('../data/train.txt')
    dev_examples = load_examples('../data/dev.txt')
    train(model, train_examples, dev_examples)

    # Load best trained model
    model = pickle.load(open('best_model.pkl', 'rb'))
    examples = load_examples('../data/test.txt')
    test(model, examples)


if __name__ == '__main__':
    main()