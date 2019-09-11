import random
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(params.input_dim, params.embed_dim)

        # dropout is used between each layer of a multi-layered RNN  as we only use a single layer, don't pass dropout
        self.gru = nn.GRU(params.embed_dim,
                          params.hidden_dim)

        self.dropout = nn.Dropout(params.dropout)

    def forward(self, source, source_length):
        # source = [source length, batch size]

        embedded = self.dropout(self.embedding(source))
        # embedded = [source length, batch size, embed dim]

        packed_embedded = pack_padded_sequence(embedded, source_length)
        packed_output, hidden = self.gru(packed_embedded)

        output, _ = pad_packed_sequence(packed_output)  # pad tokens are all-zeros
        # output = [source length, batch size, hidden dim]
        # hidden = [1, batch size, hidden dim]

        # return hidden state to initialize the first layer of the decoder
        return hidden


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(params.output_dim, params.embed_dim)

        self.gru = nn.GRU(params.embed_dim + params.hidden_dim,
                          params.hidden_dim)

        self.fc = nn.Linear(params.embed_dim + (params.hidden_dim * 2), params.output_dim)

        self.dropout = nn.Dropout(params.dropout)

    def forward(self, target, hidden, context):
        # target  = [batch size]
        # hidden  = [1, batch size, hidden dim]
        # context = [1, batch size, hidden dim]
        # re-use the same context vector returned by the encoder for every time-step in the decoder

        target = target.unsqueeze(0)
        # target = [1, batch size]

        embedded = self.dropout(self.embedding(target))
        # embedded = [1, batch size, embed dim]

        # pass current input token and context vector from the encoder to the GRU by concatenating them
        embed_con = torch.cat((embedded, context), dim=2)
        # embed_con = [1, batch size, embed dim + hidden dim]

        output, hidden = self.gru(embed_con, hidden)
        # output = [1, batch size, hidden dim]
        # hidden = [1, batch size, hidden dim]

        # addition of current input token to the linear layer means this layer can directly see what the token is,
        # without having to get this information from the hidden state

        # concatenate current input token, hidden state and context vector from the encoder together as output
        # before feeding it through the linear layer to receive our predictions
        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), dim=1)
        # output = [batch size, embed dim + (hidden dim * 2)]

        prediction = self.fc(output)
        # prediction = [batch size, output dim]

        # returns a prediction and a new hidden state
        return prediction, hidden


class Seq2SeqGRU(nn.Module):
    def __init__(self, params):
        super(Seq2SeqGRU, self).__init__()
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

        target_max_len = target.shape[0]
        batch_size = target.shape[1]

        outputs = torch.zeros(target_max_len, batch_size, self.params.output_dim).to(self.params.device)
        # outputs = [target length, batch size, output dim]

        # last hidden state of the encoder will be used as context
        context = self.encoder(source, source_length)

        # context also used as the initial hidden state of the decoder
        hidden = context

        # initial input to the decoder is <sos> tokens
        input = target[0, :]
        # input = [batch size]

        for time in range(1, target_max_len):
            output, hidden = self.decoder(input, hidden, context)
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
