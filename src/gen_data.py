'''Code to generate simple data using a predefined target function'''

import numpy as np
from matplotlib import pyplot as plt

from data_loader import save_data
from visualize import scatter_examples


def target_fn(x: list) -> int:
    '''The function to model'''
    x = np.array(x)
    if x[0] < 0:
        return 1
    return 0

    # The following is equivalent to XOR function
    if x[0] < 0 and x[1] < 0:
        return 1

    if x[0] > 0 and x[1] > 0:
        return 1
    return 0


def gen_x(count: int) -> list:
    '''
    Generate uniform random dots on 2D plane, sample range is a square centered
    at the origin with side length of 10. 
    Return a list of 2-tuples.
    '''
    vx = []
    for i in range(count):
        x = (np.random.uniform(-5, 5), np.random.uniform(-5, 5))
        vx.append(x)
    return vx


def gen_examples(count: int) -> list:
    '''Generate list of (x, y), where y is inferred using the target function'''
    vx = gen_x(count)   # Generate random x using predefined sample method
    examples = [(x, target_fn(x)) for x in vx]  # Get labels using target function
    return examples


if __name__ == '__main__':
    np.random.seed(0)
    # Generate and save train, dev, test data
    train_examples = gen_examples(800)
    save_data(train_examples, '../data/train.txt')
    dev_examples = gen_examples(200)
    save_data(dev_examples, '../data/dev.txt')
    test_examples = gen_examples(200)
    save_data(test_examples, '../data/test.txt')

    scatter_examples(train_examples)