import os
import tqdm
import torch
from torch import optim, nn
import utils
import model
import config


class SentenceTensors:
    def __init__(self, word_to_index, device):
        self.word_to_index = word_to_index
        self.device = device

    def pair_to_tensors(self, pair):
        # input data format is (input_sentence, target_sentence)
        # unknown token for ouot of vocabulary
        UNK_token = self.word_to_index['<UNK>']

        # input sentence to index
        input_sentence = pair[0]
        input_indexes = [self.word_to_index.get(word, UNK_token) for word in input_sentence.split(' ')]
        # target sentence to index
        target_sentence = pair[1]
        target_indexes = [self.word_to_index['<SOS>']]
        target_indexes.extend([self.word_to_index.get(word, UNK_token) for word in target_sentence.split(' ')])
        target_indexes.append(self.word_to_index['<EOS>'])

        # index to tensor
        input_tensor = torch.tensor(input_indexes, dtype=torch.long, device=self.device).view(-1, 1)
        target_tensor = torch.tensor(target_indexes, dtype=torch.long, device=self.device).view(-1, 1)

        return input_tensor, target_tensor


class Trainer(object):
    """Training Helper Class"""
    def __init__(self, cfg, file_dir, model, word_to_index, encoder_optimizer, decoder_optimizer, save_dir, device):
        self.cfg = cfg
        self.file = open(file_dir, 'r', errors='ignore')
        self.model = model
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.sentence_tensors = SentenceTensors(word_to_index, device)
        self.save_dir = save_dir
        self.device = device

    def data_loader(self):
        batch = []
        for i in range(self.cfg['batch_size']):
            temp = self.file.readline()
            if not temp:
                break
            else:
                batch.append(temp)
        return batch

    def train(self, get_loss, model_file=None):
        """ Train Loop """
        self.model.train() # train mode
        # self.load(model_file)
        model = self.model.to(self.device)
        global_step = 0 # global iteration steps regardless of epochs
        for e in range(self.cfg['n_epochs']):
            loss_sum = 0. # the sum of iteration losses to get average loss in every epoch
            batch = self.data_loader()
            while len(batch) == self.cfg['batch_size']:
                for i, pair in enumerate(batch):
                    self.decoder_optimizer.zero_grad()
                    self.encoder_optimizer.zero_grad()

                    input_tensor, target_tensor = self.sentence_tensors.pair_to_tensors((pair, pair))
                    encoder_outputs, decoder_outputs = model(input_tensor, target_tensor)
                    loss = get_loss(target_tensor, decoder_outputs)
                    loss_sum += loss

                    loss.backward()
                    self.decoder_optimizer.step()
                    self.encoder_optimizer.step()
                    global_step += 1

                    if global_step % self.cfg['save_steps'] == 0: # save
                    if global_step % self.cfg['save_steps'] == 0: # save
                        self.save(global_step)

                    if global_step % 1 == 0:
                        print(loss_sum / global_step)
        self.save(global_step)

    def eval(self, evaluate, model_file):
        """ Evaluation Loop """
        self.model.eval() # evaluation mode
        # self.load(model_file, None)
        model = self.model.to(self.device)

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
        torch.save(self.model.state_dict(),
            os.path.join(self.save_dir, 'model_steps_'+str(i)+'.pt'))


def main(data_dir='../data/data.txt', vocab_dir='../data/vocab.txt', embed_dir='../data/glove_common_50d.txt', save_dir='../model/181228'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = config.cfg
    word_to_index, index_to_word = utils.make_index_dict(vocab_dir)
    weight_matrix = utils.embedding_params(embed_dir, 50)
    encoder = model.EncoderRNN(
        vocab_size=len(word_to_index.keys()),
        hidden_size=cfg['hidden_size'],
        weight_matrix=weight_matrix,
        device=device)
    encoder_optimizer = optim.SGD(
        filter(lambda x: x.requires_grad, encoder.parameters()),
        lr=cfg['learning_rate'])
    decoder = model.DecoderRNN(
        hidden_size=cfg['hidden_size'],
        output_size=len(word_to_index.keys()),
        weight_matrix=weight_matrix,
        device=device)
    decoder_optimizer = optim.SGD(
        filter(lambda x: x.requires_grad, decoder.parameters()),
        lr=cfg['learning_rate'])

    seq2seq = model.Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        max_length=config.cfg['max_length'])

    trainer = Trainer(
        cfg=cfg,
        file_dir=data_dir,
        model=seq2seq,
        word_to_index=word_to_index,
        encoder_optimizer=encoder_optimizer,
        decoder_optimizer=decoder_optimizer,
        save_dir=save_dir,
        device=device)

    def get_loss(target, output):
        length = min(len(target), len(output))
        criterion = nn.NLLLoss()
        loss = 0
        for idx in range(length):
            loss += criterion(output[idx], target[idx])
        return loss

    trainer.train(get_loss)


if __name__ == '__main__':
    main()