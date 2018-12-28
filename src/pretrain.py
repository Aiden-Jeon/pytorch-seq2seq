import random
import os
import tqdm
import torch
from torch import nn, optim
import torch.nn.functional as F
import utils
import model

class SentenceTensors:
    def __init__(self, word_to_index):
        self.word_to_index = word_to_index

    def pair_to_tensors(self, pair):
        UNK_token = self.word_to_index['<UNK>']
        input_sentence = pair[0]
        input_indexes = [self.word_to_index.get(word, UNK_token) for word in input_sentence.split(' ')]

        target_sentence = pair[1]
        target_indexes = [self.word_to_index['<SOS>']]
        target_indexes.extend([self.word_to_index.get(word, UNK_token) for word in target_sentence.split(' ')])
        target_indexes.append(self.word_to_index['<EOS>'])

        input_tensor = torch.tensor(input_indexes, dtype=torch.long, device=self.device).view(-1, 1)
        target_tensor = torch.tensor(target_indexes, dtype=torch.long, device=self.device).view(-1, 1)

        return input_tensor, target_tensor


class Trainer(object):
    """Training Helper Class"""
    def __init__(self, cfg, file_dir, model, get_loss, encoder_optimizer, decoder_optimizer, save_dir, device):
        self.cfg = cfg
        self.file_dir = file_dir
        self.model = model
        self.get_loss = get_loss
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.save_dir = save_dir
        self.device = device # device name

    def train(self, get_loss, model_file=None):
        """ Train Loop """
        self.model.train() # train mode
        self.load(model_file)
        model = self.model.to(self.device)
        global_step = 0 # global iteration steps regardless of epochs
        for e in range(self.cfg.n_epochs):
            loss_sum = 0. # the sum of iteration losses to get average loss in every epoch
            # iter_bar = self.data_iter.pair_to_tensor(data) -> make data iter

            for i,  in enumerate(iter_bar):
                self.decoder_optimizer.zero_grad()
                self.encoder_optimizer.zero_grad()
                encoder_outputs, decoder_outputs = model(input_tensor, target_tensor)
                loss = get_loss(input_tensor, encoder_outputs)
                loss += get_loss(target_tensor, encoder_outputs)
                loss.backward()
                self.decoder_optimizer.step()

                global_step += 1
                loss_sum += loss.item()
                iter_bar.set_description('Iter (loss=%5.3f)'%loss.item())

                if global_step % self.cfg.save_steps == 0: # save
                    self.save(global_step)

                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum/(i+1)))
                    print('The Total Steps have been reached.')
                    self.save(global_step) # save and finish when global_steps reach total_steps
                    return

            print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum/(i+1)))
        self.save(global_step)

    def eval(self, evaluate, model_file, data_parallel=True):
        """ Evaluation Loop """
        self.model.eval() # evaluation mode
        self.load(model_file, None)
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        results = [] # prediction results
        iter_bar = tqdm(self.data_iter, desc='Iter (loss=X.XXX)')
        for batch in iter_bar:
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad(): # evaluation without gradient calculation
                accuracy, result = evaluate(model, batch) # accuracy to print
            results.append(result)

            iter_bar.set_description('Iter(acc=%5.3f)'%accuracy)
        return results

    def load(self, model_file):
        """ load saved model or pretrained transformer (a part of model) """
        print('Loading the model from', model_file)
        self.model.load_state_dict(torch.load(model_file))

    def save(self, i):
        """ save current model """
        torch.save(self.model.state_dict(), # save model object before nn.DataParallel
            os.path.join(self.save_dir, 'model_steps_'+str(i)+'.pt'))



def pair_data(data_dir):
    pair = []
    with open(data_dir, 'r') as file:
        for line in file:
            line = line.replace('\n', '')
            pair.append((line, line))
    return pair


def main(data_dir='../data/data.txt', vocab_dir='../data/vocab.txt', embed_dir='../data/glove_common_50d.txt'):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # word_to_index, index_to_word = utils.make_index_dict(vocab_dir)
    # embedding_params = utils.embedding_params(embed_dir, 50)
    # data = pair_data(data_dir)
    # trainer = Trainer(
    #     embeddings=embedding_params,
    #     max_length=10,
    #     hidden_size=50,
    #     learning_rate=0.1,
    #     teacher_forcing_ratio=0.5,
    #     word_to_index=word_to_index,
    #     device=device)
    # trainer.run(data)
    data_loader = DataLoader('../data/data.txt', 1)
    for x in data_loader:
        print(x)


if __name__ == '__main__':
    main()