import torch
import argparse
import sys

from network import RnnFactory


class Setting:
    """
    This class handles the configuration and argument parsing for training and evaluation.
    It supports different datasets (Gowalla & Foursquare) and manages GPU settings, hyperparameters,
    and other model-related parameters.
    """


    def parse(self):
        """
        Parses command-line arguments to configure training settings.
        Detects if the dataset is Foursquare and applies the appropriate defaults.
        """
        self.guess_foursquare = any(['4sq' in argv for argv in sys.argv])  # foursquare has different default args.

        parser = argparse.ArgumentParser()
        if self.guess_foursquare:
            self.parse_foursquare(parser)
        else:
            self.parse_gowalla(parser)
        self.parse_arguments(parser)
        args = parser.parse_args()

        ###### settings ######
        # training
        self.gpu = args.gpu
        self.hidden_dim = args.hidden_dim
        self.weight_decay = args.weight_decay
        self.learning_rate = args.lr
        self.epochs = args.epochs
        self.rnn_factory = RnnFactory(args.rnn)
        self.rnn = args.rnn
        self.is_lstm = self.rnn_factory.is_lstm()
        self.lambda_t = args.lambda_t
        self.lambda_s = args.lambda_s

        # data management
        self.dataset_file = './data/{}'.format(args.dataset)
        self.offset_file = './data/{}'.format(args.offset)
        self.dataset = args.dataset
        self.max_users = 0  # 0 = use all available users
        self.sequence_length = 20
        self.batch_size = args.batch_size
        self.min_checkins = 101

        # evaluation
        self.validate_epoch = args.validate_epoch
        self.report_user = args.report_user

        ### CUDA Setup ###
        self.device = torch.device('cpu') if args.gpu == -1 else torch.device('cuda', args.gpu)

    def parse_arguments(self, parser):
        """
        Defines general command-line arguments related to training and evaluation.
        """
        # training
        parser.add_argument('--gpu', default=-1, type=int, help='the gpu to use')
        parser.add_argument('--hidden-dim', default=10, type=int, help='hidden dimensions to use')
        parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
        parser.add_argument('--epochs', default=100, type=int, help='amount of epochs')
        parser.add_argument('--rnn', default='rnn', type=str, help='the GRU implementation to use: [rnn|gru|lstm]')

        # data management
        parser.add_argument('--dataset', default='checkins-gowalla.txt', type=str,
                            help='the dataset under ./data/<dataset.txt> to load')
        # parser.add_argument('--offset', default='checkins_gowalla_time_offset.txt', type=str, help='the dataset under ./data/<dataset.txt> to load')
        # evaluation
        # parser.add_argument('--validate-epoch', default=5, type=int, help='run each validation after this amount of epochs')
        parser.add_argument('--report-user', default=-1, type=int,
                            help='report every x user on evaluation (-1: ignore)')

    def parse_gowalla(self, parser):
        """
        Defines default parameters for the Gowalla dataset.
        """

        parser.add_argument('--batch-size', default=200, type=int,
                            help='amount of users to process in one pass (batching)')
        parser.add_argument('--validate-epoch', default=5, type=int,
                            help='run each validation after this amount of epochs')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=1000, type=float, help='decay factor for spatial data')
        parser.add_argument('--weight_decay', default=0.000001, type=float, help='weight decay regularization')
        parser.add_argument('--offset', default='checkins_gowalla_time_offset.txt', type=str,
                            help='the dataset under ./data/<dataset.txt> to load')

    def parse_foursquare(self, parser):
        """
        Defines default parameters for the Foursquare dataset.
        """

        parser.add_argument('--batch-size', default=512, type=int,
                            help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=100, type=float, help='decay factor for spatial data')
        parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay regularization')
        parser.add_argument('--validate-epoch', default=10, type=int,
                            help='run each validation after this amount of epochs')
        parser.add_argument('--offset', default='checkins_4sq_time_offset.txt', type=str,
                            help='the dataset under ./data/<dataset.txt> to load')

    def __str__(self):
        """
        Returns a string representation of the settings, indicating which dataset defaults were used.
        """
        return (
            'parse with foursquare default settings' if self.guess_foursquare else 'parse with gowalla default settings') + '\n' \
            + 'use device: {}'.format(self.device)