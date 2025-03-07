import torch
from torch.utils.data import DataLoader  # PyTorch DataLoader for batching data
import numpy as np
import time  # For tracking training time

# Import custom modules:
from setting import Setting  # Contains hyperparameters and command-line options
from trainer import ReplayTrainer  # Implements training and evaluation routines for REPLAY
from dataloader import PoiDataloader  # Loads and preprocesses check-in data
from dataset import Split  # Enum for specifying training/testing split
from network import create_h0_strategy  # Function to create a hidden state initialization strategy
from evaluation import Evaluation  # Module to evaluate model performance on test data

# ------------------------------------------------------------------------------
# 1. Parse settings and configurations.
# ------------------------------------------------------------------------------
setting = Setting()  # Instantiate a Setting object containing hyperparameters and configuration.
setting.parse()  # Parse command-line or config file arguments.
print(setting)  # Print current settings for verification.

# ------------------------------------------------------------------------------
# 2. Load and preprocess the data.
# ------------------------------------------------------------------------------
# Create a data loader for check-in data with user limits and minimum check-ins as specified.
poi_loader = PoiDataloader(setting.max_users, setting.min_checkins)
# Read the main dataset file and its corresponding offset file.
poi_loader.read(setting.dataset_file, setting.offset_file)
# Create a training dataset using the preprocessed data and an 80/20 split.
dataset = poi_loader.create_dataset(setting.sequence_length, setting.batch_size, Split.TRAIN)
# Wrap the dataset in a DataLoader; batch_size=1 means each DataLoader sample contains one batch.
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
# Similarly, create the test dataset (20% of the data).
dataset_test = poi_loader.create_dataset(setting.sequence_length, setting.batch_size, Split.TEST)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
# Ensure the batch size is smaller than the total number of users available.
assert setting.batch_size < poi_loader.user_count(), 'batch size must be lower than the amount of available users'

# ------------------------------------------------------------------------------
# 3. Prepare the Trainer, Hidden State Strategy, and Evaluation.
# ------------------------------------------------------------------------------
# Instantiate the ReplayTrainer, passing in lambda parameters for temporal and spatial regularizations.
trainer = ReplayTrainer(setting.lambda_t, setting.lambda_s)
# Create a strategy for initializing/resetting the RNN hidden state.
h0_strategy = create_h0_strategy(setting.hidden_dim, setting.is_lstm)
# Prepare the trainer with necessary information:
#   - Total number of POIs (locations)
#   - Total number of users
#   - Hidden dimension size
#   - RNN factory (to create the chosen RNN type)
#   - Device (CPU/GPU) for computation
trainer.prepare(poi_loader.locations(), poi_loader.user_count(), setting.hidden_dim, setting.rnn_factory,
                setting.device)
# Set up evaluation on the test dataset.
evaluation_test = Evaluation(dataset_test, dataloader_test, poi_loader.user_count(), h0_strategy, trainer, setting)
# Print a summary of the trainer and the selected RNN type.
print('{} {}'.format(trainer, setting.rnn_factory))

# ------------------------------------------------------------------------------
# 4. Set up the optimizer and learning rate scheduler.
# ------------------------------------------------------------------------------
# Use Adam optimizer on the trainer's parameters with specified learning rate and weight decay.
optimizer = torch.optim.Adam(trainer.parameters(), lr=setting.learning_rate, weight_decay=setting.weight_decay)
# Use a multi-step LR scheduler to reduce the learning rate at specific epochs.
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.2)

# ------------------------------------------------------------------------------
# 5. Training loop over epochs.
# ------------------------------------------------------------------------------
for e in range(setting.epochs):
    # Initialize the hidden state for the batch at the start of each epoch.
    h = h0_strategy.on_init(setting.batch_size, setting.device)
    # Shuffle the order of users in the dataset to ensure randomness.
    dataset.shuffle_users()

    losses = []  # To accumulate loss values for the epoch.
    epoch_start = time.time()  # Mark the epoch start time.

    # Iterate over batches from the DataLoader.
    for i, (x, t, t_slot, s, y, y_t, y_t_slot, y_s, reset_h, active_users) in enumerate(dataloader):
        # For each sequence in the batch, check if we need to reset the hidden state.
        for j, reset in enumerate(reset_h):
            if reset:
                # If using LSTM, hidden state is a tuple (h, c).
                if setting.is_lstm:
                    hc = h0_strategy.on_reset(active_users[0][j])
                    h[0][0, j] = hc[0]
                    h[1][0, j] = hc[1]
                else:
                    # Otherwise, reset the hidden state vector for that user.
                    h[0, j] = h0_strategy.on_reset(active_users[0][j])

        # Squeeze and move input tensors to the designated device.
        x = x.squeeze().to(setting.device)
        t = t.squeeze().to(setting.device)
        t_slot = t_slot.squeeze().to(setting.device)
        s = s.squeeze().to(setting.device)

        # Squeeze and move label tensors.
        y = y.squeeze().to(setting.device)
        y_t = y_t.squeeze().to(setting.device)
        y_t_slot = y_t_slot.squeeze().to(setting.device)
        y_s = y_s.squeeze().to(setting.device)
        active_users = active_users.to(setting.device)

        # Zero out gradients before the backward pass.
        optimizer.zero_grad()
        # Compute the loss for the current batch using the trainer's loss function.
        loss = trainer.loss(x, t, t_slot, s, y, y_t, y_t_slot, y_s, h, active_users)
        # Backpropagate the loss; retain_graph=True is used if the computation graph needs to be reused.
        loss.backward(retain_graph=True)
        # Record the loss value.
        losses.append(loss.item())
        # Update model parameters.
        optimizer.step()

    # Update the learning rate according to the scheduler.
    scheduler.step()
    epoch_end = time.time()  # Mark the end time of the epoch.
    print('One training need {:.2f}s'.format(epoch_end - epoch_start))

    # Print epoch metrics (loss and current learning rate).
    if (e + 1) % 1 == 0:
        epoch_loss = np.mean(losses)
        print(f'Epoch: {e + 1}/{setting.epochs}')
        print(f'Used learning rate: {scheduler.get_last_lr()[0]}')
        print(f'Avg Loss: {epoch_loss}')

    # Periodically run evaluation on the test set.
    if (e + 1) % setting.validate_epoch == 0:
        print(f'~~~ Test Set Evaluation (Epoch: {e + 1}) ~~~')
        evaluation_test.evaluate()
