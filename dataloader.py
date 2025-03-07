# dataloader.py
import os.path  # For file path checking and handling
import sys  # For exiting the program with error messages
from datetime import datetime, timedelta  # For parsing timestamps and applying time offsets

# Import the dataset class and usage enum from the dataset module.
# PoiDataset is later used to wrap the processed data for training/testing.
from dataset import PoiDataset, Usage


class PoiDataloader():
    """
    A class to load and preprocess raw check-in data for the REPLAY model.
    It reads the input file (and a corresponding offset file), filters users based on check-in counts,
    maps user and POI IDs to internal sequential indices, and organizes the data into sequences.

    This processing is critical because:
      - It ensures that only users with sufficient data (check-ins) are used.
      - It converts raw timestamps into numerical values and time slots (e.g., hour in week),
        which are later used to capture temporal regularities.
      - It builds the inputs required by REPLAY, such as user trajectories, coordinate information,
        and POI indices.
    """

    def __init__(self, max_users=0, min_checkins=0):
        # max_users: Limit on number of users to load (0 means no limit).
        # min_checkins: Minimum check-ins a user must have to be considered (e.g., 101).
        self.max_users = max_users  # 0 means no restriction.
        self.min_checkins = min_checkins  # For instance, 101 check-ins minimum.

        # Data file download instructions (see README for details):
        # https://www.dropbox.com/s/6qyrvp1epyo72xd/data.zip?dl=0

        # Mapping from original user IDs to internal sequential IDs.
        self.user2id = {}
        # Mapping from original POI IDs to internal sequential IDs.
        self.poi2id = {}

        # Lists to store processed data for each user.
        self.users = []  # List of user IDs (after filtering and mapping)
        self.times = []  # List of check-in times (in Unix seconds) for each user
        self.time_slots = []  # List of time slots (computed as weekday*24 + hour) for each check-in
        self.coords = []  # List of (latitude, longitude) tuples for each check-in
        self.locs = []  # List of POI IDs (after mapping) for each check-in

    def create_dataset(self, sequence_length, batch_size, split, usage=Usage.MAX_SEQ_LENGTH, custom_seq_count=1):
        """
        Wraps the preprocessed data into a PoiDataset, which organizes the data into sequences
        suitable for training or testing the REPLAY model.

        Args:
          sequence_length: The length of each sequence.
          batch_size: Number of sequences per batch.
          split: Indicates training or testing split (using an enum from the dataset module).
          usage: Method of handling sequence lengths (e.g., MAX_SEQ_LENGTH).
          custom_seq_count: Custom sequence count if applicable.

        Returns:
          An instance of PoiDataset initialized with the preprocessed lists.
        """
        return PoiDataset(
            self.users.copy(),
            self.times.copy(),
            self.time_slots.copy(),
            self.coords.copy(),
            self.locs.copy(),
            sequence_length,
            batch_size,
            split,
            usage,
            len(self.poi2id),  # Total number of POIs
            custom_seq_count
        )

    def user_count(self):
        """Return the total number of users loaded into the dataloader."""
        return len(self.users)

    def locations(self):
        """Return the total number of unique POIs."""
        return len(self.poi2id)

    def checkins_count(self):
        """
        Return the total number of check-ins across all users.
        This is done by summing the lengths of the check-in lists in self.locs.
        """
        count = 0
        for loc in self.locs:
            count += len(loc)
        return count

    def read(self, file, offsetfile):
        """
        Reads and processes the raw check-in data.

        Args:
          file: The main data file containing check-in records.
          offsetfile: A file containing offset values (used to adjust timestamps).

        Checks for the existence of both files; if either is missing, prints an error message and exits.
        Then, it processes the file to extract user data and POI (location) data.
        """
        if not os.path.isfile(file):
            print('[Error]: Dataset not available: {}. Please follow instructions under ./data/README.md'.format(file))
            sys.exit(1)
        if not os.path.isfile(offsetfile):
            print('[Error]: Dataset offset not available: {}. Please follow instructions under ./data/README.md'.format(
                file))
            sys.exit(1)

        # Process user and POI data.
        self.read_users(file)
        self.read_pois(file, offsetfile)

    def read_users(self, file):
        """
        Processes the data file to count check-ins per user and filter out users with too few check-ins.
        Users that meet the minimum check-ins requirement are assigned a new internal sequential ID.

        Args:
          file: Path to the main check-in data file.

        This filtering is vital since the REPLAY model relies on sufficiently long trajectories to learn temporal patterns.
        """
        f = open(file, 'r')
        lines = f.readlines()

        # Start with the user ID from the first line.
        prev_user = int(lines[0].split('\t')[0])
        visit_cnt = 0  # Count the number of check-ins for the current user.
        for i, line in enumerate(lines):
            tokens = line.strip().split('\t')
            user = int(tokens[0])
            if user == prev_user:
                # Still processing the same user; increment check-in count.
                visit_cnt += 1
            else:
                # New user encountered: Check if the previous user has enough check-ins.
                if visit_cnt >= self.min_checkins:
                    # Assign a new sequential internal ID to the user.
                    self.user2id[prev_user] = len(self.user2id)
                # Uncomment below to log discarded users:
                # else:
                #    print('discard user {}: too few checkins ({})'.format(prev_user, visit_cnt))
                prev_user = user  # Update to the new user.
                visit_cnt = 1  # Reset check-in count for the new user.
                # If a maximum number of users is set and reached, break early.
                if 0 < self.max_users <= len(self.user2id):
                    break

    def read_pois(self, file, offsetfile):
        """
        Processes the main data file and the offset file to extract detailed check-in information.

        For each check-in:
          - Converts the timestamp into Unix time.
          - Adjusts the timestamp using an offset (in minutes) to obtain a corrected time.
          - Computes a 'time slot' as (weekday * 24 + hour) to capture periodic patterns.
          - Extracts geographical coordinates.
          - Maps the original POI to a new sequential ID.
          - Groups check-in data for each user.

        Args:
          file: Path to the main check-in data file.
          offsetfile: Path to the offset file containing time adjustments.

        This function builds the lists (users, times, time_slots, coords, locs) that form the inputs to the REPLAY model.
        """
        f = open(file, 'r')
        lines = f.readlines()
        f2 = open(offsetfile, 'r')
        offsets = f2.readlines()

        # Temporary lists to accumulate check-in data for the current user.
        user_time = []  # Check-in times for the current user
        user_coord = []  # Coordinates for the current user
        user_loc = []  # POI IDs (mapped) for the current user
        user_time_slot = []  # Time slots computed for the current user

        # Get the internal ID of the first user.
        prev_user = int(lines[0].split('\t')[0])
        prev_user = self.user2id.get(prev_user)  # Map the original ID to an internal index

        # Process each check-in along with its corresponding offset.
        for i, (line, offset) in enumerate(zip(lines, offsets)):
            tokens = line.strip().split('\t')
            user = int(tokens[0])

            # If this user was filtered out (i.e., does not meet min check-ins), skip this record.
            if self.user2id.get(user) is None:
                continue
            # Map the original user ID to the internal ID.
            user = self.user2id.get(user)

            # Parse the check-in timestamp and convert it to Unix time.
            time = (datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ") - datetime(1970, 1, 1)).total_seconds()

            # Adjust the timestamp by adding the offset (offset is given in minutes).
            new_date = datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ") + timedelta(minutes=int(offset))

            # Calculate the time slot: combines the day of the week and the hour.
            # This is later used for creating smoothed timestamp embeddings.
            time_slot = new_date.weekday() * 24 + new_date.hour

            # Convert the latitude and longitude values to float and pack into a tuple.
            lat = float(tokens[2])
            long = float(tokens[3])
            coord = (lat, long)

            # Process the POI information.
            location = int(tokens[4])
            # If the POI hasn't been seen before, assign a new sequential internal ID.
            if self.poi2id.get(location) is None:
                self.poi2id[location] = len(self.poi2id)
            # Map the original POI ID to the internal ID.
            location = self.poi2id.get(location)

            # If the check-in belongs to the same user as the previous record:
            if user == prev_user:
                # Insert the new check-in data at the beginning of the list.
                # Inserting at index 0 reverses the order so that the most recent check-in comes last.
                user_time.insert(0, time)
                user_time_slot.insert(0, time_slot)
                user_coord.insert(0, coord)
                user_loc.insert(0, location)
            else:
                # When a new user is encountered, append the accumulated data for the previous user.
                self.users.append(prev_user)  # Add the previous user's internal ID
                self.times.append(user_time)  # Their list of check-in times
                self.time_slots.append(user_time_slot)  # Their list of computed time slots
                self.coords.append(user_coord)  # Their list of coordinates
                self.locs.append(user_loc)  # Their list of POI IDs (internal)

                # Restart accumulation for the new user.
                prev_user = user
                user_time = [time]
                user_time_slot = [time_slot]
                user_coord = [coord]
                user_loc = [location]

        # Append the data for the final user after the loop ends.
        self.users.append(prev_user)
        self.times.append(user_time)
        self.time_slots.append(user_time_slot)
        self.coords.append(user_coord)
        self.locs.append(user_loc)
