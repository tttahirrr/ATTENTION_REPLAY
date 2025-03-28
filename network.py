import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from enum import Enum
from math import pi


class Rnn(Enum):
    ''' The available RNN units '''

    RNN = 0
    GRU = 1
    LSTM = 2

    @staticmethod
    def from_string(name):
        """ Converts a string to its corresponding RNN type. """
        if name == 'rnn':
            return Rnn.RNN
        if name == 'gru':
            return Rnn.GRU
        if name == 'lstm':
            return Rnn.LSTM
        raise ValueError('{} not supported in --rnn'.format(name))


class RnnFactory():
    ''' Creates the desired RNN unit. '''

    def __init__(self, rnn_type_str):
        self.rnn_type = Rnn.from_string(rnn_type_str)

    def __str__(self):
        """ Returns a string representation of the chosen RNN type. """
        if self.rnn_type == Rnn.RNN:
            return 'Use pytorch RNN implementation.'
        if self.rnn_type == Rnn.GRU:
            return 'Use pytorch GRU implementation.'
        if self.rnn_type == Rnn.LSTM:
            return 'Use pytorch LSTM implementation.'

    def is_lstm(self):
        """ Checks if the selected RNN type is LSTM. """
        return self.rnn_type in [Rnn.LSTM]

    def create(self, hidden_size):
        """ Creates an instance of the chosen RNN type with the specified hidden size. """
        if self.rnn_type == Rnn.RNN:
            return nn.RNN(hidden_size, hidden_size)
        if self.rnn_type == Rnn.GRU:
            return nn.GRU(hidden_size, hidden_size)
        if self.rnn_type == Rnn.LSTM:
            return nn.LSTM(hidden_size, hidden_size)


class User_Week_Distribution(nn.Module):
    """ Models a Gaussian distribution over weekly time slots to learn temporal patterns. """

    def __init__(self, stamp_num):
        super().__init__()
        self.stamp_num = stamp_num
        self.sigma = nn.Parameter(torch.ones(self.stamp_num).view(self.stamp_num, 1))

    def forward(self, x):
        """ Computes the Gaussian distribution for a given time input. """
        self.sigma.data = torch.abs(self.sigma.data)
        learned_weight = 1 / torch.sqrt(2 * pi * (self.sigma ** 2)) * torch.exp(-(x ** 2) / (2 * (self.sigma ** 2)))
        sum = torch.sum(learned_weight, dim=1, keepdim=True)
        return learned_weight / sum
        # return 1/torch.sqrt(2*pi*(self.sigma**2))*torch.exp((-x**2)/(2*(self.sigma**2)))


# class User_Day_Distribution(nn.Module):
#     def __init__(self,stamp_num):
#         super().__init__()
#         self.stamp_num=stamp_num
#         self.sigma=nn.Parameter(torch.ones(self.stamp_num).view(self.stamp_num,1))

# def forward(self,x):
#     # sigma=self.user_day_sigma.index_select(0,active_user.view(-1)).view(user_len,-1)
#     return 1/torch.sqrt(2*pi*(self.sigma**2))*torch.exp((-x**2)/(2*(self.sigma**2)))

class FlashbackAttention(nn.Module):
    def __init__(self, hidden_size):
        super(FlashbackAttention, self).__init__()
        # Learnable projection to compute a joint representation
        self.attn_proj = nn.Linear(hidden_size * 2, hidden_size)
        # Vector for scoring (could be multi-head, but this is a single-head version)
        self.attn_score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, current_hidden, past_hidden):
        """
        current_hidden: Tensor of shape (batch, hidden_size) representing the current state.
        past_hidden: Tensor of shape (batch, seq_len, hidden_size) representing historical hidden states.
        Returns:
           attended: Tensor of shape (batch, hidden_size) — a weighted combination of past_hidden.
           attn_weights: Tensor of shape (batch, seq_len) with the attention scores.
        """
        batch_size, seq_len, hidden_size = past_hidden.size()

        # Expand current_hidden to shape (batch, seq_len, hidden_size)
        current_expanded = current_hidden.unsqueeze(1).expand(-1, seq_len, -1)

        # Concatenate along hidden dimension: (batch, seq_len, 2*hidden_size)
        concat = torch.cat((past_hidden, current_expanded), dim=-1)

        # Compute a joint representation and then a scalar score for each time step
        energy = torch.tanh(self.attn_proj(concat))  # (batch, seq_len, hidden_size)
        scores = self.attn_score(energy).squeeze(-1)  # (batch, seq_len)

        # Normalize scores to get weights
        attn_weights = F.softmax(scores, dim=-1)  # (batch, seq_len)

        # Compute weighted sum of past_hidden states
        attended = torch.bmm(attn_weights.unsqueeze(1), past_hidden).squeeze(1)  # (batch, hidden_size)
        return attended, attn_weights


class REPLAY(nn.Module):
    """
    Core neural network model for location prediction.
    Uses embeddings, an RNN, and a custom temporal/spatial weighting mechanism with attention-enhanced flashback.
    """

    def __init__(self, input_size, user_count, hidden_size, f_t, f_s, rnn_factory, week, day, week_weight_index,
                 day_weight_index):
        super().__init__()
        self.input_size = input_size
        self.user_count = user_count
        self.hidden_size = hidden_size
        self.f_t = f_t  # function for computing temporal weight
        self.f_s = f_s  # function for computing spatial weight
        self.week_matrix = week  # 168 x 168 matrix
        self.week_weight_index = week_weight_index

        self.encoder = nn.Embedding(input_size, hidden_size)  # location embedding
        self.user_encoder = nn.Embedding(user_count, hidden_size)  # user embedding
        self.week_encoder = nn.Embedding(24 * 7, hidden_size // 2)
        self.rnn = rnn_factory.create(hidden_size)
        # fcpt projects the concatenated POI and temporal embeddings to the RNN hidden size
        self.fcpt = nn.Linear(2 * hidden_size - hidden_size // 2, hidden_size)
        # Update: the final fc layer now takes concatenated features of dimension:
        # current hidden state (hidden_size) + attended context (hidden_size) + user embedding (hidden_size) + future time embedding (hidden_size//2)
        # Total = 3 * hidden_size + hidden_size//2
        self.fc = nn.Linear(3 * hidden_size + hidden_size // 2, input_size)
        self.week_distribution = User_Week_Distribution(168)

        # New attention module for flashback
        self.flashback_attn = FlashbackAttention(hidden_size)

    def forward(self, x, t, t_slot, s, y_t, y_t_slot, y_s, h, active_user):
        """
        Forward pass of the REPLAY model with attention-enhanced flashback.
        Returns predictions for each time step in the input sequence.

        :param x: Input locations (POIs) [seq_len x user_len].
        :param t: Timestamps of check-ins.
        :param t_slot: Time slot indices.
        :param s: Spatial coordinates.
        :param y_t: Next time-step timestamps.
        :param y_t_slot: Next time-step time slot indices.
        :param y_s: Next time-step spatial coordinates.
        :param h: Hidden state of the RNN.
        :param active_user: IDs of active users in the batch.
        :return: predictions (shape: [seq_len, user_len, input_size]), updated hidden state.
        """
        seq_len, user_len = x.size()

        # ------------------Week-based Temporal Embedding------------------
        week_weight = self.week_distribution(self.week_weight_index).view(168, 168)
        new_week_weight1 = week_weight.index_select(0, t_slot.view(-1)).view(seq_len, user_len, 168, 1)
        new_week_weight2 = week_weight.index_select(0, y_t_slot.view(-1)).view(seq_len, user_len, 168, 1)

        w_t1 = self.week_matrix.index_select(0, t_slot.view(-1)).view(seq_len, user_len, -1)
        w_t1 = self.week_encoder(w_t1).permute(0, 1, 3, 2)  # shape: (seq_len, user_len, hidden//2, 168)
        w_t1 = torch.matmul(w_t1, new_week_weight1).squeeze()  # shape: (seq_len, user_len, hidden//2)
        t_emb1 = w_t1

        w_t2 = self.week_matrix.index_select(0, y_t_slot.view(-1)).view(seq_len, user_len, -1)
        w_t2 = self.week_encoder(w_t2).permute(0, 1, 3, 2)  # shape: (seq_len, user_len, hidden//2, 168)
        w_t2 = torch.matmul(w_t2, new_week_weight2).squeeze()  # shape: (seq_len, user_len, hidden//2)
        t_emb2 = w_t2

        # ------------------Embedding and RNN Computation------------------
        x_emb = self.encoder(x)  # (seq_len, user_len, hidden_size)
        # Combine POI embedding with temporal embedding (t_emb1)
        poi_time = self.fcpt(torch.cat((x_emb, t_emb1), dim=-1))  # (seq_len, user_len, hidden_size)
        out, h = self.rnn(poi_time, h)  # out: (seq_len, user_len, hidden_size)

        # ------------------Prepare Fixed Future Time and User Embedding------------------
        # We use the fixed query time from the last time step of t_emb2 for all predictions.
        fixed_t_future = t_emb2[-1]  # ideally (user_len, hidden_size//2)
        if fixed_t_future.dim() > 2:
            fixed_t_future = fixed_t_future.contiguous().view(fixed_t_future.size(0), -1)

        # Ensure active_user is a 1D tensor of indices.
        if active_user.dim() > 1:
            active_user = active_user.squeeze()
        user_emb = self.user_encoder(active_user)  # (user_len, hidden_size)

        # ------------------Time-Distributed Prediction------------------
        predictions = []
        # For each time step i, compute prediction using flashback on past hidden states
        for i in range(seq_len):
            # current hidden state at time step i: (user_len, hidden_size)
            h_current_i = out[i]
            # Past hidden states up to time step i: (i+1, user_len, hidden_size) -> transpose to (user_len, i+1, hidden_size)
            past_hidden_i = out[:i + 1].transpose(0, 1)
            # Apply flashback attention: using current state and its past hidden states
            attended_i, _ = self.flashback_attn(h_current_i, past_hidden_i)
            # Combine current state, attended context, user embedding, and fixed future time embedding
            combined_i = torch.cat((h_current_i, attended_i, user_emb, fixed_t_future), dim=-1)
            # Predict next POI for time step i (for all users)
            pred_i = self.fc(combined_i)  # (user_len, input_size)
            predictions.append(pred_i.unsqueeze(0))  # add time dimension
        # Stack predictions: (seq_len, user_len, input_size)
        predictions = torch.cat(predictions, dim=0)

        return predictions, h


def create_h0_strategy(hidden_size, is_lstm):
    """ Factory function to create the appropriate initialization strategy for hidden states. """
    if is_lstm:
        return LstmStrategy(hidden_size, FixNoiseStrategy(hidden_size), FixNoiseStrategy(hidden_size))
    else:
        return FixNoiseStrategy(hidden_size)


class H0Strategy():
    """ Base class for different hidden state initialization strategies. """

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def on_init(self, user_len, device):
        pass

    def on_reset(self, user):
        pass

    def on_reset_test(self, user, device):
        return self.on_reset(user)


class FixNoiseStrategy(H0Strategy):
    ''' use fixed normal noise as initialization '''

    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        mu = 0
        sd = 1 / self.hidden_size
        self.h0 = torch.randn(self.hidden_size, requires_grad=False) * sd + mu

    def on_init(self, user_len, device):
        hs = []
        for i in range(user_len):
            hs.append(self.h0)
        return torch.stack(hs, dim=0).view(1, user_len, self.hidden_size).to(device)

    def on_reset(self, user):
        return self.h0


class LstmStrategy(H0Strategy):
    ''' creates h0 and c0 using the inner strategy '''

    def __init__(self, hidden_size, h_strategy, c_strategy):
        super(LstmStrategy, self).__init__(hidden_size)
        self.h_strategy = h_strategy
        self.c_strategy = c_strategy

    def on_init(self, user_len, device):
        h = self.h_strategy.on_init(user_len, device)
        c = self.c_strategy.on_init(user_len, device)
        return (h, c)

    def on_reset(self, user):
        h = self.h_strategy.on_reset(user)
        c = self.c_strategy.on_reset(user)
        return (h, c)