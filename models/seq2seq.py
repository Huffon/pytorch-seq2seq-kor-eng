import random
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(config.kor_vocab_size, config.embed_dim)

        self.lstm = nn.LSTM(config.embed_dim,
                            config.hidden_dim,
                            config.n_layer,
                            dropout=config.dropout)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input):
        # input = [input length, batch size]

        embedded = self.embedding(input)
        # embedded = [input length, batch size, embed dim]

        output, (hidden, cell) = self.lstm(embedded)

        # output = [input length, batch size, hidden dim]

        # hidden = [num layers, batch size, hidden dim]
        # cell   = [num layers, batch size, hidden dim]
        # [forward_layer_0, forward_layer_1, ..., forward_layer_n]

        # return hidden and cell states to initialize the first layer of the decoder

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(config.eng_vocab_size, config.embed_dim)

        self.lstm = nn.LSTM(config.embed_dim,
                            config.hidden_dim,
                            config.n_layer,
                            dropout=config.dropout)

        self.fc = nn.Linear(config.hidden_dim, config.eng_vocab_size)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input, hidden, cell):
        # input  = [batch size]
        # hidden = [num_layers, batch size, hidden dim]
        # cell   = [num_layers, batch size, hidden dim]

        # add additional dimension to let shape be same as the input dimension of Embedding layer
        input = input.unsqueeze(0)
        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, embed dim]

        # first initialize weights with the encoder's hidden and cell states
        # after that using weights of the decoder's hidden and cell states from last time step
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        # output = [1, batch size, hidden dim]
        # hidden = [num layers, batch size, hidden dim]
        # [forward_layer_0, forward_layer_1, ..., forward_layer_n]
        # cell   = [num layers, batch size, hidden dim]

        assert output.squeeze(0).shape == hidden[-1, :, :].shape

        prediction = self.fc(output.squeeze(0))
        # prediction = [batch size, output dim]

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, config):
        super(Seq2Seq, self).__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, source, target, teacher_forcing=0.5):
        # source = [source input length, batch size]
        # target = [target output length, batch size]

        # the length of the output shouldn't exceeds the lengths of target sentences
        target_max_len = target.shape[0]
        # batch size changes dynamically, so takes batch size from the batch of target sentences
        batch_size = target.shape[1]

        # define outputs tensor used to store outputs of the decoder
        outputs = torch.zeros(target_max_len, batch_size, self.config.eng_vocab_size).to(self.config.device)

        # last hidden and cell states of the encoder is used to initialize the initial hidden state of the decoder
        hidden, cell = self.encoder(source)

        # initial input to the decoder is <sos> tokens
        input = target[0, :]
        # input = [batch size]

        for time in range(1, target_max_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            # output contains the predicted results which has the size of output dim (eng_vocab_size)
            # output = [batch size, output dim]

            # store the output of each time step to the 'outputs' tensor
            outputs[time] = output

            # calculates boolean flag whether to use teacher forcing
            teacher_force = random.random() < teacher_forcing

            # pick the token which has the largest value from the output
            # output.max(1) is a tuple containing (the largest tokens'soutput probability, index of that token)
            top1 = output.max(1)[1]
            # top1 = [batch size]

            # if use teacher forcing, next input token is the ground-truth token
            # if we don't, next input token is the predicted token
            input = (target[time] if teacher_force else top1)

        return outputs

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
