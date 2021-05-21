import time
import os
import mne
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import brainflow
from brainflow.board_shim import BoardIds, BrainFlowInputParams, BoardShim
from brainflow.data_filter import DataFilter

# main directory
main_dir = os.getcwd()

# OpenBCI Cyton board
cyton_board_ID = 0
print("\n" + str(BoardIds(cyton_board_ID)))  

# serial connection
serial_port = 'COM3'
params = BrainFlowInputParams()
params.serial_port = serial_port

# create board
board = BoardShim(board_id=cyton_board_ID, input_params=params)

# board attributes
board_name = board.get_device_name(board_id=board.board_id)
channel_names = board.get_eeg_names(board_id=board.board_id)
samp_freq = board.get_sampling_rate(board_id=board.board_id)  # 250 Hz
eeg_channel_idx = board.get_eeg_channels(board_id=board.board_id)  # EEG channel indices (1-8)

# prepare the board for data stream
board.prepare_session()
print("Session ready.")

# start data stream
board.start_stream()
print("Streaming data...")

# initial timestamp
timestamp = time.time()
# number of seconds to record
time.sleep(15)

# get all data
data = board.get_board_data()  

# stop the data stream
board.stop_stream()
board.release_session()

# eeg data directory
eeg_dir = os.path.join(main_dir, "raw_eeg_data")

# save recorded eeg data
filename = str(timestamp) + '.csv'
os.chdir(eeg_dir)
DataFilter.write_file(data, filename, 'w')

# read data from saved file
read_data = DataFilter.read_file(filename)
read_df = pd.DataFrame(np.transpose(read_data))
os.chdir(main_dir)
filepath = os.path.join(eeg_dir, filename)

all_raw_data = np.loadtxt(filepath, delimiter=',')
raw_eeg_data = np.transpose(all_raw_data[:, eeg_channel_idx])
print(np.shape(raw_eeg_data))
scaled_raw_eeg_data = raw_eeg_data/1000000  # uV to V

# create the info structure needed by MNE
info = mne.create_info(channel_names, samp_freq, ch_types='eeg')
# create Raw object
#raw = mne.io.RawArray(raw_eeg_data, info)
raw = mne.io.RawArray(scaled_raw_eeg_data, info)

# notch filtering
notch_freqs = (60, 120)
#scaled_notched = raw.copy().notch_filter(freqs=notch_freqs, notch_widths=0.5)
scaled_notched = raw.copy().notch_filter(freqs=notch_freqs)
#scaled_filtered = scaled_notched.copy().filter_data(h_freq=0.5)

# plot eeg data
scaled_notched.plot(block=True)
scaled_notched.plot_psd(fmax=samp_freq/2, average=True)

# plot eeg data
#raw.plot(block=True)
#raw.plot_psd(fmax=samp_freq/2)