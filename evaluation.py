import torch
import numpy as np


class Evaluation:
    """
    The Evaluation class handles model evaluation on the test set.
    It computes metrics such as recall@1, recall@5, recall@10, and Mean Average Precision (MAP)
    across all users.

    It works by iterating through the dataloader batches, generating predictions using the trainer's evaluate method,
    and then comparing those predictions against the ground truth labels.
    """

    def __init__(self, dataset, dataloader, user_count, h0_strategy, trainer, setting):
        """
        Initialize the Evaluation object.

        Args:
            dataset: The dataset instance (e.g., PoiDataset) used for evaluation.
            dataloader: DataLoader that yields batches from the dataset.
            user_count: Total number of users in the dataset.
            h0_strategy: Strategy for initializing/resetting hidden states of the RNN.
            trainer: The trainer instance that wraps the model and its evaluation logic.
            setting: Configuration settings, including device info, batch size, and other hyperparameters.
        """
        self.dataset = dataset
        self.dataloader = dataloader
        self.user_count = user_count
        self.h0_strategy = h0_strategy
        self.trainer = trainer
        self.setting = setting

    def evaluate(self):
        """
        Evaluates the model on the test set.

        Workflow:
          1. Reset the dataset to start from the beginning.
          2. Initialize hidden states using the provided h0_strategy.
          3. Iterate through each batch in the dataloader without tracking gradients.
          4. For each batch, if a new user's sequence starts (reset flag is True), reset its hidden state.
          5. Move all data to the specified device (CPU/GPU).
          6. Use the trainer to get model outputs and updated hidden states.
          7. For each sample in the batch, extract the top-10 predictions and compute per-user metrics:
             - Recall@1, Recall@5, Recall@10, and precision for MAP.
          8. Aggregate metrics over all users and print out final evaluation numbers.
        """
        # Reset dataset pointers for a new evaluation run.
        self.dataset.reset()
        # Initialize the hidden state for the batch (e.g., zeros or via a specific strategy).
        h = self.h0_strategy.on_init(self.setting.batch_size, self.setting.device)

        # Disable gradient tracking for evaluation.
        with torch.no_grad():
            # Initialize counters for global metrics.
            iter_cnt = 0
            recall1 = 0
            recall5 = 0
            recall10 = 0
            average_precision = 0.0

            # Initialize per-user arrays to aggregate metrics.
            u_iter_cnt = np.zeros(self.user_count)
            u_recall1 = np.zeros(self.user_count)
            u_recall5 = np.zeros(self.user_count)
            u_recall10 = np.zeros(self.user_count)
            u_average_precision = np.zeros(self.user_count)
            reset_count = torch.zeros(self.user_count)  # Count how many times each user's sequence resets

            # Iterate over batches from the dataloader.
            for i, (x, t, t_slot, s, y, y_t, y_t_slot, y_s, reset_h, active_users) in enumerate(self.dataloader):
                # Squeeze the active_users tensor to remove unnecessary dimensions.
                active_users = active_users.squeeze()
                # For each sample in the batch, check if the sequence is a reset (start of a new user trajectory)
                for j, reset in enumerate(reset_h):
                    if reset:
                        # If using LSTM, hidden state is a tuple (h, c); else, just h.
                        if self.setting.is_lstm:
                            hc = self.h0_strategy.on_reset_test(active_users[j], self.setting.device)
                            h[0][0, j] = hc[0]
                            h[1][0, j] = hc[1]
                        else:
                            h[0, j] = self.h0_strategy.on_reset_test(active_users[j], self.setting.device)
                        # Increment reset count for that user (to avoid evaluating multiple sequences from the same user).
                        reset_count[active_users[j]] += 1

                # Move input data to the appropriate device. Squeeze in case batch size is 1.
                x = x.squeeze().to(self.setting.device)
                t = t.squeeze().to(self.setting.device)
                t_slot = t_slot.squeeze().to(self.setting.device)
                s = s.squeeze().to(self.setting.device)

                # Move labels to device (if needed) and squeeze.
                y = y.squeeze()
                y_t = y_t.squeeze().to(self.setting.device)
                y_t_slot = y_t_slot.squeeze().to(self.setting.device)
                y_s = y_s.squeeze().to(self.setting.device)
                active_users = active_users.to(self.setting.device)

                # Call the trainer's evaluate function to get predictions and updated hidden state.
                # 'out' contains the prediction scores (votes for each POI) for every time step.
                out, h = self.trainer.evaluate(x, t, t_slot, s, y_t, y_t_slot, y_s, h, active_users)

                # Process each sample in the batch.
                for j in range(self.setting.batch_size):
                    # o is a matrix of scores for the j-th sample (across all sequence time steps).
                    o = out[j]

                    # Convert scores to a NumPy array to work with numpy functions.
                    o_n = o.cpu().detach().numpy()
                    # Get indices of the top 10 predictions for each time step.
                    ind = np.argpartition(o_n, -10, axis=1)[:, -10:]

                    # Get ground truth labels for this sample.
                    y_j = y[:, j]

                    # For each time step in the sequence:
                    for k in range(len(y_j)):
                        # Skip evaluation for a user if they have been reset more than once (to avoid duplicate evaluation).
                        if reset_count[active_users[j]] > 1:
                            continue
                        # For the k-th time step, sort the top 10 predictions in descending order.
                        ind_k = ind[k]
                        r = ind_k[np.argsort(-o_n[k, ind_k], axis=0)]
                        r = torch.tensor(r)
                        t_val = y_j[k]  # The true POI index.

                        # Compute precision for MAP: count how many scores are higher than the true label's score.
                        r_kj = o_n[k, :]
                        t_score = r_kj[t_val]
                        upper = np.where(r_kj > t_score)[0]
                        precision = 1.0 / (1 + len(upper))

                        # Accumulate per-user metrics.
                        u_iter_cnt[active_users[j]] += 1
                        u_recall1[active_users[j]] += t_val in r[:1]  # Check if true label is in top-1.
                        u_recall5[active_users[j]] += t_val in r[:5]  # Check if true label is in top-5.
                        u_recall10[active_users[j]] += t_val in r[:10]  # Check if true label is in top-10.
                        u_average_precision[active_users[j]] += precision

            # Formatter for printing metrics with 8 decimal places.
            formatter = "{0:.8f}"
            # Aggregate metrics over all users.
            for j in range(self.user_count):
                iter_cnt += u_iter_cnt[j]
                recall1 += u_recall1[j]
                recall5 += u_recall5[j]
                recall10 += u_recall10[j]
                average_precision += u_average_precision[j]

                # Optionally report intermediate metrics for a subset of users.
                if self.setting.report_user > 0 and (j + 1) % self.setting.report_user == 0:
                    print('Report user', j, 'preds:', u_iter_cnt[j],
                          'recall@1', formatter.format(u_recall1[j] / u_iter_cnt[j]),
                          'MAP', formatter.format(u_average_precision[j] / u_iter_cnt[j]), sep='\t')

            # Finally, print global evaluation metrics.
            print('recall@1:', formatter.format(recall1 / iter_cnt))
            print('recall@5:', formatter.format(recall5 / iter_cnt))
            print('recall@10:', formatter.format(recall10 / iter_cnt))
            print('MAP', formatter.format(average_precision / iter_cnt))
            print('predictions:', iter_cnt)
