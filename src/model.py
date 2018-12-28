import torch
from torch import nn
import torch.nn.functional as F
import random

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, weight_matrix, device):
        super(EncoderRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        # embedding layer
        self.embed_layer = nn.Embedding(vocab_size, hidden_size)
        self.embed_layer.load_state_dict({'weight': torch.from_numpy(weight_matrix)})   # call pre-trained matrix
        self.embed_layer.requires_grad = False  # do not train embedding layer
        self.LSTM = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input, hidden):
        input_embed = self.embed_layer(input).view(1, 1, -1)
        output, hidden = self.LSTM(input_embed, hidden)
        return output, hidden

    def init_hidden(self):
        # init_hidden state is zero
        hidden = (torch.zeros(1, 1, self.hidden_size, device=self.device),
                 torch.zeros(1, 1, self.hidden_size, device=self.device))
        return hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, weight_matrix, device):
        super(DecoderRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size

        self.embed_layer = nn.Embedding(output_size, hidden_size)
        self.embed_layer.load_state_dict({'weight': torch.from_numpy(weight_matrix)})
        self.embed_layer.requires_grad = False
        self.LSTM = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embed_layer(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.LSTM(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        hidden = (torch.zeros(1, 1, self.hidden_size, device=self.device),
                 torch.zeros(1, 1, self.hidden_size, device=self.device))
        return hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, max_length, EOS_index=1):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.max_length = max_length
        self.EOS_index = EOS_index

    def forward(self, input_tensor, target_tensor, teacher_forcing_ratio=0.5):
        input_length = min(input_tensor.size(0), self.max_length)
        target_length = target_tensor.size(0)
        # Encoder
        encoder_hidden = self.encoder.init_hidden()
        encoder_outputs = []  # to hold encoder output
        for idx in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[idx], encoder_hidden)
            encoder_outputs.append(encoder_output)

        # Decoder
        decoder_input = target_tensor[0]
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        # Not to use teacher_forcing_ratio = 0.0
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for idx in range(1, target_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
                decoder_input = target_tensor[idx]  # Teacher forcing
                decoder_outputs.append(decoder_output)
        else:
            # Without teacher forcing: use its own predictions as the next input
            for idx in range(1, target_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                decoder_outputs.append(decoder_output)
                if decoder_input.item() == self.EOS_index:
                    break

        return encoder_outputs, decoder_outputs
