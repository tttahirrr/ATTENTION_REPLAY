import os.path
import sys

from datetime import datetime, timedelta

from dataset import PoiDataset, Usage


class PoiDataloader():
    """
    Class to handle the loading and preprocessing of POI (Point of Interest) check-in data.
    It processes user check-ins, timestamps, locations, and converts them into datasets for training/testing.
    """
    def __init__(self, max_users=0, min_checkins=0):
        """
        Initializes the dataloader with optional constraints on max users and min check-ins per user.

        :param max_users: Maximum number of users to load (0 means no limit).
        :param min_checkins: Minimum number of check-ins required for a user to be included.
        """
        self.max_users = max_users  # 0
        self.min_checkins = min_checkins  # 101

        self.user2id = {}
        self.poi2id = {}

        self.users = []
        self.times = []
        self.time_slots = []
        self.coords = []
        self.locs = []

    def create_dataset(self, sequence_length, batch_size, split, usage=Usage.MAX_SEQ_LENGTH, custom_seq_count=1):
        """
        Creates a dataset object using the stored check-in data.

        :param sequence_length: Number of check-ins in a sequence.
        :param batch_size: Number of sequences per batch.
        :param split: Defines whether the dataset is for training or testing.
        :param usage: Determines how sequences are generated (default: max sequence length).
        :param custom_seq_count: Number of custom sequences to generate per user.
        :return: A PoiDataset object with formatted check-in sequences.
        """
        return PoiDataset(self.users.copy(),
                          self.times.copy(),
                          self.time_slots.copy(),
                          self.coords.copy(),
                          self.locs.copy(),
                          sequence_length,
                          batch_size,
                          split,
                          usage,
                          len(self.poi2id),
                          custom_seq_count)

    def user_count(self):
        """Returns the total number of users in the dataset."""
        return len(self.users)

    def locations(self):
        """Returns the total number of unique locations (POIs) in the dataset."""
        return len(self.poi2id)

    def checkins_count(self):
        """Returns the total number of check-ins across all users."""
        count = 0
        for loc in self.locs:
            count += len(loc)
        return count

    def read(self, file, offsetfile):
        """
        Reads the dataset files and processes user check-ins.

        :param file: Path to the main dataset file.
        :param offsetfile: Path to the file containing time offsets for check-ins.
        """
        if not os.path.isfile(file):
            print('[Error]: Dataset not available: {}. Please follow instructions under ./data/README.md'.format(file))
            sys.exit(1)
        if not os.path.isfile(offsetfile):
            print('[Error]: Dataset offset not available: {}. Please follow instructions under ./data/README.md'.format(
                file))
            sys.exit(1)

        # Process users and check-ins
        self.read_users(file)

        self.read_pois(file, offsetfile)

    def read_users(self, file):
        """
        Reads user data from the dataset and filters based on min_checkins.

        :param file: Path to the dataset file.
        """
        f = open(file, 'r')
        lines = f.readlines()

        prev_user = int(lines[0].split('\t')[0])
        visit_cnt = 0
        for i, line in enumerate(lines):
            tokens = line.strip().split('\t')
            user = int(tokens[0])
            if user == prev_user:
                visit_cnt += 1
            else:
                if visit_cnt >= self.min_checkins:
                    self.user2id[prev_user] = len(self.user2id)
                # else:
                #    print('discard user {}: to few checkins ({})'.format(prev_user, visit_cnt))
                prev_user = user
                visit_cnt = 1
                if 0 < self.max_users <= len(self.user2id):
                    break  # restrict to max users

    def read_pois(self, file, offsetfile):
        """
        Reads POI (location) data from the dataset and maps it to internal indices.

        :param file: Path to the dataset file.
        :param offsetfile: Path to the time offset file.
        """
        f = open(file, 'r')
        lines = f.readlines()
        f2 = open(offsetfile, 'r')
        offsets = f2.readlines()
        # store location ids
        user_time = []
        user_coord = []
        user_loc = []
        user_time_slot = []

        prev_user = int(lines[0].split('\t')[0])
        prev_user = self.user2id.get(prev_user)  # from 0
        for i, (line, offset) in enumerate(zip(lines, offsets)):
            # for i, line in enumerate(lines):
            tokens = line.strip().split('\t')
            user = int(tokens[0])
            if self.user2id.get(user) is None:
                continue  # user is not of interest(inactive user)
            user = self.user2id.get(user)  # from 0

            time = (datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ") - datetime(1970, 1,
                                                                                  1)).total_seconds()  # unix seconds

            new_date = datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ") + timedelta(minutes=int(offset))
            # print('new date:',new_date)
            # print('no offset:',datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ"))
            # print('offset:',int(offset)/60)
            # print('week day:',new_date.weekday())
            # print('hour:',new_date.hour)
            # print('-------------')
            # if i == 15:
            #     break
            time_slot = new_date.weekday() * 24 + new_date.hour
            lat = float(tokens[2])
            long = float(tokens[3])
            coord = (lat, long)

            location = int(tokens[4])
            if self.poi2id.get(location) is None:
                self.poi2id[location] = len(self.poi2id)

            location = self.poi2id.get(location)  # from 0

            if user == prev_user:

                user_time.insert(0, time)  # insert in front!
                user_time_slot.insert(0, time_slot)
                user_coord.insert(0, coord)
                user_loc.insert(0, location)
            else:
                self.users.append(prev_user)  # 添加用户
                self.times.append(user_time)  # 添加列表
                self.time_slots.append(user_time_slot)
                self.coords.append(user_coord)
                self.locs.append(user_loc)
                # print(len(user_time) == len(user_time_slot) == len(user_loc) == len(user_coord))
                # restart:
                prev_user = user
                user_time = [time]
                user_time_slot = [time_slot]
                user_coord = [coord]
                user_loc = [location]

        self.users.append(prev_user)
        self.times.append(user_time)
        self.time_slots.append(user_time_slot)
        self.coords.append(user_coord)
        self.locs.append(user_loc)
