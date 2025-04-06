import torch
import torch.nn as nn
import numpy as np
from network import REPLAY


class ReplayTrainer():
    """
    This class handles the training process for the REPLAY model.
    It defines the model, loss function, training/evaluation steps, and helper functions.
    """
    def __init__(self, lambda_t, lambda_s):
        """
        Initializes the trainer with hyperparameters.

        :param lambda_t: Temporal decay factor.
        :param lambda_s: Spatial decay factor.
        """
        self.lambda_t = lambda_t
        self.lambda_s = lambda_s
        self.max_grad_norm = 1.0  # Add gradient clipping threshold

    def __str__(self):
        """ Returns a string representation of the trainer. """
        return 'Use REPLAY training.'

    def parameters(self):
        """ Returns the model's parameters for optimization. """
        return self.model.parameters()

    def generate_tensor_of_distribution(self, time):
        """
        Generates a tensor representing a cyclic distribution of time indices.

        :param time: Total time steps (e.g., 168 for weekly data, 24 for daily data).
        :return: Tensor of time indices, cyclically shifted.
        """
        list1 = []
        temp = [i for i in range(time)]
        for i in range(time):
            if i == time // 2:
                list1.append(temp)
            elif i < time // 2:
                list1.append(temp[-(time // 2 - i):] + temp[:-(time // 2 - i)])
            else:
                list1.append(temp[(i - time // 2):] + temp[:(i - time // 2)])
        return torch.tensor(list1)

    def prepare(self, loc_count, user_count, hidden_size, gru_factory, device):
        """
        Prepares the REPLAY model for training.

        :param loc_count: Number of unique locations (POIs).
        :param user_count: Number of users.
        :param hidden_size: Hidden size of the model.
        :param gru_factory: Factory for creating the RNN (RNN, GRU, or LSTM).
        :param device: Device to run the model on (CPU/GPU).
        """
        f_t = lambda delta_t, user_len: ((torch.cos(delta_t * 2 * np.pi / 86400) + 1) / 2) * torch.exp(
            -(delta_t / 86400 * self.lambda_t))  # hover cosine + exp decay
        f_s = lambda delta_s, user_len: torch.exp(-(delta_s * self.lambda_s))  # exp decay
        self.loc_count = loc_count
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.week = self.generate_tensor_of_distribution(168).to(device)
        self.day = self.generate_tensor_of_distribution(24).to(device)
        self.week_weight_index = torch.tensor([x - 84 for x in range(168)]).repeat(168, 1).to(device)
        self.day_weight_index = torch.tensor([x - 12 for x in range(24)]).repeat(24, 1).to(device)
        self.model = REPLAY(loc_count, user_count, hidden_size, f_t, f_s, gru_factory, self.week, self.day,
                            self.week_weight_index, self.day_weight_index).to(device)

        # Add gradient clipping
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

    def evaluate(self, x, t, t_slot, s, y_t, y_t_slot, y_s, h, active_users):
        """
        Runs the model in evaluation mode and returns predictions.

        :param x: Input locations (POIs).
        :param t: Timestamps of check-ins.
        :param t_slot: Time slot indices.
        :param s: Spatial coordinates.
        :param y_t: Next time-step timestamps.
        :param y_t_slot: Next time-step time slot indices.
        :param y_s: Next time-step spatial coordinates.
        :param h: Hidden state of the RNN.
        :param active_users: IDs of active users in the batch.
        :return: Model predictions and updated hidden state.
        """
        self.model.eval()
        out, h = self.model(x, t, t_slot, s, y_t, y_t_slot, y_s, h, active_users)
        out_t = out.transpose(0, 1)
        return out_t, h

    def loss(self, x, t, t_slot, s, y, y_t, y_t_slot, y_s, h, active_users):
        """
        Computes the loss for a batch of input sequences.

        :param x: Input locations (POIs).
        :param t: Timestamps of check-ins.
        :param t_slot: Time slot indices.
        :param s: Spatial coordinates.
        :param y: Ground truth next locations.
        :param y_t: Next time-step timestamps.
        :param y_t_slot: Next time-step time slot indices.
        :param y_s: Next time-step spatial coordinates.
        :param h: Hidden state of the RNN.
        :param active_users: IDs of active users in the batch.
        :return: Computed loss value.
        """
        self.model.train()
        out, h = self.model(x, t, t_slot, s, y_t, y_t_slot, y_s, h, active_users)
        out = out.view(-1, self.loc_count)
        y = y.view(-1)
        l = self.cross_entropy_loss(out, y)
        
        # Add L2 regularization for attention weights
        l2_reg = 0.01 * sum(p.pow(2.0).sum() for p in self.model.parameters())
        l = l + l2_reg
        
        return l

    def train_step(self, x, t, t_slot, s, y, y_t, y_t_slot, y_s, h, active_users):
        """Performs a single training step with gradient clipping"""
        self.optimizer.zero_grad()
        loss = self.loss(x, t, t_slot, s, y, y_t, y_t_slot, y_s, h, active_users)
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        return loss.item()