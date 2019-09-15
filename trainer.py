import time
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim

from utils import init_weights, init_weights_gru, init_weights_attention, epoch_time
from models.seq2seq import Seq2Seq
from models.seq2seq_gru import Seq2SeqGRU
from models.seq2seq_attention import Seq2SeqAttention

random.seed(32)
torch.manual_seed(32)
torch.backends.cudnn.deterministic = True


class Trainer:
    def __init__(self, params, mode, train_iter=None, valid_iter=None, test_iter=None):
        self.params = params

        # Train mode
        if mode == 'train':
            self.train_iter = train_iter
            self.valid_iter = valid_iter

        # Test mode
        else:
            self.test_iter = test_iter

        model_type = {
            'seq2seq': Seq2Seq,
            'seq2seq_gru': Seq2SeqGRU,
            'seq2seq_attention': Seq2SeqAttention,
        }

        self.model = model_type[self.params.model](self.params)
        self.model.to(self.params.device)

        # SGD updates all parameters with the same learning rate
        # Adam adapts learning rate for each parameter
        optim_type = {
            'SGD': optim.SGD(self.model.parameters(), lr=self.params.lr),
            'Adam': optim.Adam(self.model.parameters()),
        }

        self.optimizer = optim_type[self.params.optim]

        # CrossEntropyLoss calculates both the log-softmax as well as the negative log-likelihood
        # when calculate the loss, padding token should be ignored because it's not related to the prediction
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.params.pad_idx)
        self.criterion.to(self.params.device)

    def train(self):
        print(f'The model has {self.model.count_parameters():,} trainable parameters')

        best_valid_loss = float('inf')

        # apply the appropriate initialization method for the model
        if self.params.model == 'seq2seq':
            self.model.apply(init_weights)
        elif self.params.model == 'seq2seq_gru':
            self.model.apply(init_weights_gru)
        elif self.params.model == 'seq2seq_attention':
            self.model.apply(init_weights_attention)

        print(self.model)

        for epoch in range(self.params.num_epoch):
            self.model.train()

            epoch_loss = 0
            start_time = time.time()

            for batch in self.train_iter:
                # For each batch, first zero the gradients
                self.optimizer.zero_grad()
                sources, sources_lengths = batch.kor
                targets = batch.eng

                predictions = self.model(sources, sources_lengths, targets)
                # targets     = [target length, batch size]
                # predictions = [target length, batch size, output dim]

                # flatten the ground-truth and predictions since CrossEntropyLoss takes 2D predictions with 1D targets
                # +) in this process, we don't use 0-th token, since it is <sos> token
                targets = targets[1:].view(-1)
                predictions = predictions[1:].view(-1, predictions.shape[-1])

                # targets = [(target sentence length - 1) * batch size]
                # predictions = [(target sentence length - 1) * batch size, output dim]

                loss = self.criterion(predictions, targets)

                # clip the gradients to prevent the model from exploding gradient
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip)

                loss.backward()
                self.optimizer.step()

                # 'item' method is used to extract a scalar from a tensor which only contains a single value.
                epoch_loss += loss.item()

            train_loss = epoch_loss / len(self.train_iter)
            valid_loss = self.evaluate()

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), self.params.save_model)

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\tVal. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f}')

    def evaluate(self):
        epoch_loss = 0

        self.model.eval()

        with torch.no_grad():
            for batch in self.valid_iter:
                sources, sources_lengths = batch.kor
                targets = batch.eng

                # when validates or test the model, we shouldn't use teacher forcing
                predictions = self.model(sources, sources_lengths, targets, 0)

                predictions = predictions[1:].view(-1, predictions.shape[-1])
                targets = targets[1:].view(-1)

                loss = self.criterion(predictions, targets)

                epoch_loss += loss.item()

        return epoch_loss / len(self.valid_iter)

    def inference(self):
        epoch_loss = 0

        self.model.load_state_dict(torch.load(self.params.save_model))
        self.model.eval()

        with torch.no_grad():
            for batch in self.test_iter:
                sources, sources_lengths = batch.kor
                targets = batch.eng

                predictions = self.model(sources, sources_lengths, targets, 0)

                predictions = predictions[1:].view(-1, predictions.shape[-1])
                targets = targets[1:].view(-1)

                loss = self.criterion(predictions, targets)

                epoch_loss += loss.item()

        test_loss = epoch_loss / len(self.test_iter)
        print(f'Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f}')
