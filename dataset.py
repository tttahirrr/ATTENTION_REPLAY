import random
from enum import Enum
import torch
from torch.utils.data import Dataset


# Define enums to distinguish between training and testing splits.
class Split(Enum):
    TRAIN = 0
    TEST = 1


# Define usage modes for sequence handling.
class Usage(Enum):
    MIN_SEQ_LENGTH = 0  # Use the minimum sequence count across users.
    MAX_SEQ_LENGTH = 1  # Use the maximum sequence count across users.
    CUSTOM = 2  # Use a custom number of sequences.


class PoiDataset(Dataset):
    """
    PoiDataset organizes the preprocessed check-in data into sequences.
    It splits each user’s trajectory into input sequences (e.g., locations, times)
    and corresponding labels (next check-ins), for use in training/testing the REPLAY model.

    Key ideas:
    - It prepares both the input features and the target labels.
    - It supports different splitting strategies (e.g., training vs. testing based on an 80/20 split).
    - It arranges data into batches for sequential processing by the RNN (with flashback mechanism).
    """

    def reset(self):
        """
        Reset the dataset's internal state for an epoch.
        It initializes pointers to the active users and their current sequence indices.
        """
        self.next_user_idx = 0  # Pointer for selecting the next user from permutation.
        self.active_users = []  # List of users currently in the batch.
        self.active_user_seq = []  # List of the current sequence index for each active user.
        self.user_permutation = []  # A random permutation of all users.

        # Initialize active users for the first batch.
        for i in range(self.batch_size):
            self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
            self.active_users.append(i)
            self.active_user_seq.append(0)
        # Create a permutation of all users.
        for i in range(len(self.users)):
            self.user_permutation.append(i)

    def shuffle_users(self):
        """
        Shuffle the order of users to ensure a random sampling order across epochs.
        Resets active users and sequence indices accordingly.
        """
        random.shuffle(self.user_permutation)
        self.next_user_idx = 0
        self.active_users = []
        self.active_user_seq = []
        for i in range(self.batch_size):
            self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
            self.active_users.append(self.user_permutation[i])
            self.active_user_seq.append(0)

    def __init__(self, users, times, time_slots, coords, locs,
                 sequence_length, batch_size, split, usage, loc_count, custom_seq_count):
        """
        Initialize the dataset with preprocessed data.

        Args:
          users: List of user indices.
          times: List of check-in times per user.
          time_slots: List of computed time slots (e.g., weekday*24 + hour) per check-in.
          coords: List of geographic coordinates per check-in.
          locs: List of POI indices per user.
          sequence_length: Length of each input sequence.
          batch_size: Number of sequences per batch.
          split: Training or testing split (using Split enum).
          usage: Mode for splitting sequences (MIN_SEQ_LENGTH, MAX_SEQ_LENGTH, or CUSTOM).
          loc_count: Total number of unique POIs.
          custom_seq_count: If using CUSTOM mode, number of sequences per user.

        This constructor prepares various lists that will later be segmented into
        sequences (for both inputs and labels) that the model uses for training/testing.
        """
        self.users = users
        self.locs = locs
        self.times = times
        self.time_slots = time_slots
        self.coords = coords

        # These lists will store the labels (i.e., next POI info) and their associated features.
        self.labels = []
        self.lbl_times = []
        self.lbl_time_slots = []
        self.lbl_coords = []

        # Lists to store the final sequences.
        self.sequences = []
        self.sequences_times = []
        self.sequences_time_slots = []
        self.sequences_coords = []

        self.sequences_labels = []
        self.sequences_lbl_times = []
        self.sequences_lbl_time_slots = []
        self.sequences_lbl_coords = []

        self.sequences_count = []  # Count of sequences per user.
        self.Ps = []  # Placeholder (could be used for additional processing).
        # Qs is initialized as a tensor with shape (number of POIs, 1), containing POI indices.
        self.Qs = torch.zeros(loc_count, 1)
        self.usage = usage
        self.batch_size = batch_size
        self.loc_count = loc_count
        self.custom_seq_count = custom_seq_count

        # Initialize training state.
        self.reset()

        # Set Qs values as the POI indices (from 0 to loc_count-1).
        for i in range(loc_count):
            self.Qs[i, 0] = i

            # For each user, separate the trajectory into input features and labels.
        for i, loc in enumerate(locs):
            # For each user's trajectory, use all but the last check-in as input.
            self.locs[i] = loc[:-1]
            # The labels are the next check-in for each input check-in.
            self.labels.append(loc[1:])
            self.lbl_times.append(self.times[i][1:])
            self.lbl_time_slots.append(self.time_slots[i][1:])
            self.lbl_coords.append(self.coords[i][1:])

            # Remove the last check-in from the input sequences.
            self.times[i] = self.times[i][:-1]
            self.time_slots[i] = self.time_slots[i][:-1]
            self.coords[i] = self.coords[i][:-1]

        # Split the sequences further into fixed-length sequences.
        for i, (time, time_slot, coord, loc, label, lbl_time, lbl_time_slot, lbl_coord) in enumerate(
                zip(self.times, self.time_slots, self.coords, self.locs,
                    self.labels, self.lbl_times, self.lbl_time_slots, self.lbl_coords)):
            # Define the split threshold (80% for training, 20% for testing).
            train_thr = int(len(loc) * 0.8)
            if split == Split.TRAIN:
                self.locs[i] = loc[:train_thr]
                self.times[i] = time[:train_thr]
                self.time_slots[i] = time_slot[:train_thr]
                self.coords[i] = coord[:train_thr]

                self.labels[i] = label[:train_thr]
                self.lbl_times[i] = lbl_time[:train_thr]
                self.lbl_time_slots[i] = lbl_time_slot[:train_thr]
                self.lbl_coords[i] = lbl_coord[:train_thr]

            if split == Split.TEST:
                self.locs[i] = loc[train_thr:]
                self.times[i] = time[train_thr:]
                self.time_slots[i] = time_slot[train_thr:]
                self.coords[i] = coord[train_thr:]

                self.labels[i] = label[train_thr:]
                self.lbl_times[i] = lbl_time[train_thr:]
                self.lbl_time_slots[i] = lbl_time_slot[train_thr:]
                self.lbl_coords[i] = lbl_coord[train_thr:]

        # Split the location sequences into fixed-length sequences.
        self.max_seq_count = 0  # Maximum number of sequences across users.
        self.min_seq_count = 10000000  # A large number to be minimized.
        self.capacity = 0  # Total number of sequences.
        for i, (time, time_slot, coord, loc, label, lbl_time, lbl_time_slot, lbl_coord) in enumerate(
                zip(self.times, self.time_slots, self.coords, self.locs,
                    self.labels, self.lbl_times, self.lbl_time_slots, self.lbl_coords)):
            # Number of sequences that can be formed from this user's data.
            seq_count = len(loc) // sequence_length
            # Ensure there is at least one complete sequence.
            assert seq_count > 0, 'fix seq-length and min-checkins in order to have at least one test sequence in a 80/20 split!'
            seqs = []  # To store input sequences for this user.
            seq_times = []  # To store corresponding timestamp sequences.
            seq_time_slots = []  # To store corresponding time slot sequences.
            seq_coords = []  # To store corresponding coordinate sequences.

            seq_lbls = []  # To store label sequences (next check-ins).
            seq_lbl_times = []  # To store label timestamps.
            seq_lbl_time_slots = []  # To store label time slots.
            seq_lbl_coords = []  # To store label coordinates.

            # Split the trajectory into consecutive sequences.
            for j in range(seq_count):
                start = j * sequence_length
                end = (j + 1) * sequence_length
                seqs.append(loc[start:end])
                seq_times.append(time[start:end])
                seq_time_slots.append(time_slot[start:end])
                seq_coords.append(coord[start:end])

                seq_lbls.append(label[start:end])
                seq_lbl_times.append(lbl_time[start:end])
                seq_lbl_time_slots.append(lbl_time_slot[start:end])
                seq_lbl_coords.append(lbl_coord[start:end])

            # Store the sequences for this user.
            self.sequences.append(seqs)
            self.sequences_times.append(seq_times)
            self.sequences_time_slots.append(seq_time_slots)
            self.sequences_coords.append(seq_coords)

            self.sequences_labels.append(seq_lbls)
            self.sequences_lbl_times.append(seq_lbl_times)
            self.sequences_lbl_time_slots.append(seq_lbl_time_slots)
            self.sequences_lbl_coords.append(seq_lbl_coords)

            self.sequences_count.append(seq_count)
            self.capacity += seq_count
            self.max_seq_count = max(self.max_seq_count, seq_count)
            self.min_seq_count = min(self.min_seq_count, seq_count)

        # Print statistics to help monitor data loading.
        if self.usage == Usage.MIN_SEQ_LENGTH:
            print(split, 'load', len(users), 'users with min_seq_count', self.min_seq_count, 'batches:', self.__len__())

        if self.usage == Usage.MAX_SEQ_LENGTH:
            print(split, 'load', len(users), 'users with max_seq_count', self.max_seq_count, 'batches:', self.__len__())
        if self.usage == Usage.CUSTOM:
            print(split, 'load', len(users), 'users with custom_seq_count', self.custom_seq_count, 'Batches:',
                  self.__len__())

    def sequences_by_user(self, idx):
        """
        Return all sequences associated with a given user.
        This is useful for inspecting a user's check-in trajectory in detail.
        """
        return self.sequences[idx]

    def __len__(self):
        """
        Compute the total number of available batches.
        The calculation depends on the usage mode (MIN_SEQ_LENGTH, MAX_SEQ_LENGTH, or CUSTOM).
        """
        if self.usage == Usage.MIN_SEQ_LENGTH:
            # Minimum sequence count multiplied by the number of user batches.
            return self.min_seq_count * (len(self.users) // self.batch_size)
        if self.usage == Usage.MAX_SEQ_LENGTH:
            # Estimated capacity divided by batch size.
            estimated = self.capacity // self.batch_size
            return max(self.max_seq_count, estimated)
        if self.usage == Usage.CUSTOM:
            return self.custom_seq_count * (len(self.users) // self.batch_size)
        raise ValueError()

    def __getitem__(self, idx):
        """
        Retrieve one batch of data.

        For each element in the batch:
          - Retrieve the sequence of input data (POI indices, timestamps, time slots, coordinates).
          - Retrieve the corresponding labels (next check-ins) and their features.
          - Update the user’s pointer so that next time we get a new sequence.

        The method also handles rotating to the next user when a user’s available sequences are exhausted.
        Returns:
          - x: Tensor of input POI indices.
          - t: Tensor of input timestamps.
          - t_slot: Tensor of input time slots.
          - s: Tensor of input coordinates.
          - y: Tensor of label POI indices.
          - y_t: Tensor of label timestamps.
          - y_t_slot: Tensor of label time slots.
          - y_s: Tensor of label coordinates.
          - reset_h: List of booleans indicating if a sequence is the start of a new user's data.
          - active_users: Tensor of active user indices for this batch.
        """
        seqs = []
        times = []
        time_slots = []
        coords = []
        lbls = []
        lbl_times = []
        lbl_time_slots = []
        lbl_coords = []
        reset_h = []  # Flags to indicate if a user’s sequence is being reset.

        # Iterate over each element in the batch.
        for i in range(self.batch_size):
            # Get the current active user for this batch element.
            i_user = self.active_users[i]
            # j is the current sequence index for this user.
            j = self.active_user_seq[i]
            max_j = self.sequences_count[i_user]

            # Adjust maximum sequence count based on the usage mode.
            if self.usage == Usage.MIN_SEQ_LENGTH:
                max_j = self.min_seq_count
            if self.usage == Usage.CUSTOM:
                max_j = min(max_j, self.custom_seq_count)
            if j >= max_j:
                # If no more sequences for this user, rotate to the next user.
                i_user = self.user_permutation[self.next_user_idx]
                j = 0
                self.active_users[i] = i_user
                self.active_user_seq[i] = j
                self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
                # Ensure the next user is not already in the active batch.
                while self.user_permutation[self.next_user_idx] in self.active_users:
                    self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
            # Mark whether this is the start of a new sequence for the user.
            reset_h.append(j == 0)
            # Append the input sequence and corresponding features as tensors.
            seqs.append(torch.tensor(self.sequences[i_user][j]))
            times.append(torch.tensor(self.sequences_times[i_user][j]))
            time_slots.append(torch.tensor(self.sequences_time_slots[i_user][j]))
            coords.append(torch.tensor(self.sequences_coords[i_user][j]))

            # Append the label sequence and corresponding features.
            lbls.append(torch.tensor(self.sequences_labels[i_user][j]))
            lbl_times.append(torch.tensor(self.sequences_lbl_times[i_user][j]))
            lbl_time_slots.append(torch.tensor(self.sequences_lbl_time_slots[i_user][j]))
            lbl_coords.append(torch.tensor(self.sequences_lbl_coords[i_user][j]))

            # Advance the sequence pointer for this user.
            self.active_user_seq[i] += 1

        # Stack tensors along the new batch dimension.
        x = torch.stack(seqs, dim=1)
        t = torch.stack(times, dim=1)
        t_slot = torch.stack(time_slots, dim=1)
        s = torch.stack(coords, dim=1)

        y = torch.stack(lbls, dim=1)
        y_t = torch.stack(lbl_times, dim=1)
        y_t_slot = torch.stack(lbl_time_slots, dim=1)
        y_s = torch.stack(lbl_coords, dim=1)
        return x, t, t_slot, s, y, y_t, y_t_slot, y_s, reset_h, torch.tensor(self.active_users)
