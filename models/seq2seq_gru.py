import random
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(params.input_dim, params.embed_dim)

        # the dropout is used between each layer of a multi-layered RNN
        # as we only use a single layer, don't pass the dropout as an argument
        self.gru = nn.GRU(params.embed_dim,
                          params.hidden_dim)

        self.dropout = nn.Dropout(params.dropout)

    def forward(self, input):
        # input = [input length, batch size]

        embedded = self.dropout(self.embedding(input))
        # embedded = [input length, batch size, embed dim]

        output, hidden = self.gru(embedded)
        # output = [input length, batch size, hidden dim]
        # hidden = [1, batch size, hidden dim]

        # return hidden state to initialize the first layer of the decoder
        return hidden


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(params.output_dim, params.embed_dim)

        self.gru = nn.GRU(params.embed_dim + params.hidden_dim,
                          params.hidden_dim)

        self.fc = nn.Linear(params.embed_dim + params.hidden_dim * 2, params.output_dim)

        self.dropout = nn.Dropout(params.dropout)

    def forward(self, input, hidden, context):
        # input  = [batch size]
        # hidden = [1, batch size, hidden dim]
        # context = [1, batch size, hidden dim]
        # re-use the same context vector returned by the encoder for every time-step in the decoder

        # add additional dimension to make the shape of the input same as the input dimension of Embedding layer
        input = input.unsqueeze(0)
        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, embed dim]

        # pass current input token and context vector from the encoder to the GRU by concatenating them
        embed_con = torch.cat((embedded, context), dim=2)
        # embed_con = [1, batch size, embed dim + hidden dim]

        output, hidden = self.gru(embed_con, hidden)

        # output = [1, batch size, hidden dim]
        # hidden = [1, batch size, hidden dim]

        # addition of current input token to the linear layer means this layer can directly see what the token is,
        # without having to get this information from the hidden state.

        # concatenate current input token, hidden state and context vector from the encoder together as output
        # before feeding it through the linear layer to receive our predictions
        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), dim=1)
        # output = [batch size, embed dim + hidden dim * 2]

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

    def forward(self, source, target, teacher_forcing=0.5):
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

        # the length of the output shouldn't exceeds the lengths of target sentences
        target_max_len = target.shape[0]
        # batch size changes dynamically, so updates the batch size each time step
        batch_size = target.shape[1]

        # define 'outputs' tensor used to store each time step's output ot the decoder
        outputs = torch.zeros(target_max_len, batch_size, self.params.output_dim).to(self.params.device)
        # outputs = [target length, batch size, output dim]

        # last hidden state of the encoder will be used as context
        context = self.encoder(source)

        # context also used as the initial hidden state of the decoder
        hidden = context

        # initial input to the decoder is <sos> tokens, naming 'output' to use generically
        output = target[0, :]
        # output = [batch size]

        for time in range(1, target_max_len):
            output, hidden = self.decoder(output, hidden, context)
            # output contains the predicted results which has the size of output dim
            # output = [batch size, output dim]

            # store the output of each time step to the 'outputs' tensor
            outputs[time] = output

            # calculates boolean flag whether to use teacher forcing
            teacher_force = random.random() < teacher_forcing

            # pick the token which has the largest value from each time step's output
            # output.max(1) is a tuple containing (the largest tokens's probability, index of that token)
            top1 = output.max(1)[1]
            # top1 = [batch size]

            # if use teacher forcing, next input token is the ground-truth token
            # if we don't, next input token is the predicted token. naming 'output' to use generically
            output = (target[time] if teacher_force else top1)

            # during inference time, when encounters <eos> token, return generated outputs
            if inference and output.item() == self.params.eos_idx:
                return outputs[:time]

        return outputs

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
