import numpy as np


def make_index_dict(file_dir):
    word_to_index = {'<SOS>': 0, '<EOS>': 1, '<UNK>': 2}
    index_to_word = {0: '<SOS>', 1: '<EOS>', 2: '<UNK>'}
    with open(file_dir, 'r') as file:
        for idx, line in enumerate(file):
            line = line.replace('\n', '')
            word_to_index[line] = idx+3
            index_to_word[idx+3] = line
    return word_to_index, index_to_word


def embedding_params(file_dir, dimension):
    with open(file_dir, 'r') as file:
        values = []
        for line in file:
            value = line.split(' ')[1:]
            value = list(map(float, value))
            values.append(value)
    embed_params = np.zeros(shape=[len(values)+3, dimension])
    for idx, value in enumerate(values):
        embed_params[idx+3] = values[idx]
    return embed_params


# if __name__ == '__main__':
#     a = embedding_params('../data/glove_common_50d.txt', 50)
#     print(a['weight'][:10])