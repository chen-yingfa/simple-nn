def load_examples(filename):
    data = []
    with open(filename, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip().split()
            if len(line) == 0:
                break
            x = [float(e) for e in line[:-1]]
            y = int(line[-1])
            data.append((x, y))
    return data


def save_data(examples: list, filename: str):
    '''
    Save examples to file, where each line correspond to an example, each line
    consists of k+1 space-separated numbers, where the first k numbers are the
    elements of x, the last number is the label (integer).
    ...

    `examples`: [(x, y), ...], a list of tuples where each tuple is an example.
    `filename`: the file to save to.
    '''
    with open(filename, 'w', encoding='utf8') as f:
        for x, y in examples:
            for elem in x:
                f.write(str(elem) + ' ')
            f.write(str(y))
            f.write('\n')