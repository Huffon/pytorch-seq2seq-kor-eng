import random
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(params.input_dim, params.embed_dim)

        self.lstm = nn.LSTM(params.embed_dim,
                            params.hidden_dim,
                            params.n_layer,
                            dropout=params.dropout)

        self.dropout = nn.Dropout(params.dropout)

    def forward(self, source, source_length):
        # input = [source length, batch size]

        embedded = self.embedding(source)
        # embedded = [source length, batch size, embed dim]

        packed_embedded = pack_padded_sequence(embedded, source_length)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        output, _ = pad_packed_sequence(packed_output)   # pad tokens are all-zeros
        # output = [source length, batch size, hidden dim]

        # hidden = [num layers, batch size, hidden dim]
        # cell   = [num layers, batch size, hidden dim]
        # [forward_layer_0, forward_layer_1, ..., forward_layer_n]

        # return hidden and cell states to initialize the first layer of the decoder
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(params.output_dim, params.embed_dim)

        self.lstm = nn.LSTM(params.embed_dim,
                            params.hidden_dim,
                            params.n_layer,
                            dropout=params.dropout)

        self.fc = nn.Linear(params.hidden_dim, params.output_dim)

        self.dropout = nn.Dropout(params.dropout)

    def forward(self, target, hidden, cell):
        # target  = [batch size]
        # hidden  = [num_layers, batch size, hidden dim]
        # cell    = [num_layers, batch size, hidden dim]

        target = target.unsqueeze(0)
        # target = [1, batch size]

        embedded = self.dropout(self.embedding(target))
        # embedded = [1, batch size, embed dim]

        # at first initialize weights with the encoder's last hidden and cell states
        # after that using weights of the decoder's previous hidden and cell states
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        # output = [1, batch size, hidden dim]

        # hidden = [num layers, batch size, hidden dim]
        # cell   = [num layers, batch size, hidden dim]
        # [forward_layer_0, forward_layer_1, ..., forward_layer_n]

        assert output.squeeze(0).shape == hidden[-1, :, :].shape

        prediction = self.fc(output.squeeze(0))
        # prediction = [batch size, output dim]

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, params):
        super(Seq2Seq, self).__init__()
        self.params = params
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, source, source_length, target, teacher_forcing=0.5):
        # source = [source length, batch size]
        # target = [target length, batch size]

        # if target is None, check whether teacher_forcing is zero
        if target is None:
            assert teacher_forcing == 0, "Must be zero during inference"

            # makes inference flag True and defines dummy target sentences with max length as 100
            inference = True
            target = torch.zeros((100, source.shape[1])).long().fill_(self.params.sos_idx).to(self.params.device)
            # target = [100, 1] filled with <sos> tokens
        else:
            inference = False

        # the length of the output shouldn't exceed the length of target sentences
        target_max_len = target.shape[0]
        # batch size changes dynamically, so updates the batch size each time step
        batch_size = target.shape[1]

        # define 'outputs' tensor used to store each time step's output ot the decoder
        outputs = torch.zeros(target_max_len, batch_size, self.params.output_dim).to(self.params.device)
        # outputs = [target length, batch size, output dim]

        # last hidden and cell states of the encoder is used to initialize the initial hidden state of the decoder
        hidden, cell = self.encoder(source, source_length)

        # initial input to the decoder is <sos> tokens
        input = target[0, :]
        # input = [batch size]

        for time in range(1, target_max_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            # output contains the predicted results which has the size of output dim
            # output = [batch size, output dim]

            # store the output of each time step at the 'outputs' tensor
            outputs[time] = output

            # calculates boolean flag whether to use teacher forcing
            teacher_force = random.random() < teacher_forcing

            # pick the token which has the largest value from each time step's output
            # output.max(1) is a tuple containing (the largest tokens's probability, index of that token)
            top1 = output.max(1)[1]
            # top1 = [batch size]

            # if use teacher forcing, next input token is the ground-truth token
            # if we don't, next input token is the predicted token. naming 'output' to use generically
            input = (target[time] if teacher_force else top1)

            # during inference time, when encounters <eos> token, return generated outputs
            if inference and input.item() == self.params.eos_idx:
                return outputs[:time]

        return outputs

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
