import time
import torch
import torch.nn as nn
import torch.optim as optim

from utils import init_weights, epoch_time
from models.seq2seq import Seq2Seq


class Trainer:
    def __init__(self, config, pad_idx, train_iter=None, valid_iter=None, test_iter=None):
        self.config = config
        self.pad_idx = pad_idx

        # Train mode
        if self.config.mode == 'train':
            self.train_iter = train_iter
            self.valid_iter = valid_iter

        # Test mode
        else:
            self.test_iter = test_iter

        model_type = {
            'seq2seq': Seq2Seq(self.config),
        }

        self.model = model_type[self.config.model]
        self.model.to(self.config.device)

        # SGD updates all parameters with the same learning rate
        # Adam adapts learning rate for each parameter
        optim_type = {
            'SGD': optim.SGD(self.model.parameters(), lr=self.config.lr),
            'Adam': optim.Adam(self.model.parameters()),
        }

        self.optimizer = optim_type[self.config.optim]

        # CrossEntropyLoss calculates both the log softmax as well as the negative log-likelihood
        # when calculate the loss, padding token should be ignored because it's not related to the prediction
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        self.criterion.to(self.config.device)

    def train(self):
        print(f'The model has {self.model.count_parameters():,} trainable parameters')

        best_valid_loss = float('inf')

        self.model.train()
        # Seq2Seq model initializes weights with uniform distribution range from -0.08 to 0.08
        self.model.apply(init_weights)

        print(self.model)

        for epoch in range(self.config.num_epoch):
            epoch_loss = 0
            epoch_acc = 0

            start_time = time.time()

            for batch in self.train_iter:
                # For each batch, first zero the gradients
                self.optimizer.zero_grad()

                sources, targets = batch.kor, batch.eng
                predictions = self.model(sources, targets)
                # targets = [target sentence length, batch size]
                # predictions = [target sentence length, batch size, output dim]

                # we should flatten the ground-truth and predictions tensors
                # because CrossEntropyLoss takes 2D inputs(predictions) with 1D targets
                # +) in this process, we don't use 0-th token, since it is <sos> token
                targets = targets[1:].view(-1)
                predictions = predictions[1:].view(-1, predictions.shape[-1])

                # targets = [(target sentence length - 1) * batch size]
                # predictions = [(target sentence length - 1) * batch size, output dim]

                loss = self.criterion(predictions, targets)

                # clip the gradients to prevent the model from exploding gradient
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)

                loss.backward()
                self.optimizer.step()

                # 'item' method is used to extract a scalar from a tensor which only contains a single value.
                epoch_loss += loss.item()

            train_loss = epoch_loss / len(self.train_iter)
            train_acc = epoch_acc / len(self.train_iter)

            valid_loss, valid_acc = self.evaluate()

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), self.config.save_model)

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\tVal. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}%')

    def evaluate(self):
        epoch_loss = 0
        epoch_acc = 0

        self.model.eval()

        with torch.no_grad():
            for batch in self.valid_iter:
                sources, targets = batch.kor, batch.eng

                # when validates or test the model, we shouldn't use teacher forcing
                predictions = self.model(sources, targets, 0)

                predictions = predictions[1:].view(-1, predictions.shape[-1])
                targets = targets[1:].view(-1)

                loss = self.criterion(predictions, targets)

                epoch_loss += loss.item()

        return epoch_loss / len(self.valid_iter), epoch_acc / len(self.valid_iter)

    def inference(self):
        epoch_loss = 0
        epoch_acc = 0

        self.model.load_state_dict(torch.load(self.config.save_model))
        self.model.eval()

        with torch.no_grad():
            for batch in self.test_iter:
                sources, targets = batch.kor, batch.eng
                predictions = self.model(sources, targets, 0)

                predictions = predictions[1:].view(-1, predictions.shape[-1])
                targets = targets[1:].view(-1)

                loss = self.criterion(predictions, targets)

                epoch_loss += loss.item()

        test_loss = epoch_loss / len(self.test_iter)
        test_acc = epoch_acc / len(self.test_iter)
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
