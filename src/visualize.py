'''For plotting examples that are dots on 2D plane'''
from matplotlib import pyplot as plt


def scatter_x(vx: list):
    '''
    Scatterplot a list of tuples.
    `vx` is a list of tuples, each representing a position on 2D plane.
    '''
    x, y = zip(*vx)
    plt.scatter(x, y)


def scatter_examples(examples: list, xlim: tuple=None, ylim: tuple=None,
                     add_line: bool=False):
    '''Scatterplot list of examples'''
    # Plot
    # Divide into positive and negative examples
    pos = [x for (x, y) in examples if y == 1]
    neg = [x for (x, y) in examples if y == 0]

    # Make sure there are same number of positive and negative samples
    print(f'Number of positive samples: {len(pos)}')
    print(f'Number of negative samples: {len(neg)}')
    print('Plotting...')
    scatter_x(pos)
    scatter_x(neg)

    # Options settings of the appearance of the plot
    if xlim:
        plt.xlim(*xlim)
        plt.ylim(*ylim)
    if add_line:
        plt.plot([-1, 1], [-1, 1], 'r--')
    plt.show()


if __name__ == '__main__':
    # For debugging
    from data_loader import load_examples
    examples = load_examples('../data/dev.txt')
    scatter_examples(examples)
