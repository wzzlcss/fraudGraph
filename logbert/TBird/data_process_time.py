import sys
sys.path.append('../')

import os
import pandas as pd
import numpy as np
from logparser import Spell, Drain
from tqdm import tqdm

import torch



tqdm.pandas()
pd.options.mode.chained_assignment = None  # default='warn'

data_dir = os.path.expanduser("../dataset/tbird/")
output_dir = "../output/tbird/sliding_window_60_60_1.0/"
raw_log_file = "Thunderbird.log"
sample_log_file = "Thunderbird_20M.log"
sample_window_size = 2*10**7
sample_step_size = 10**4
window_name = ''
log_file = sample_log_file

parser_type = 'drain'
#mins
window_size = 1
step_size = 0.5
train_ratio = 6000

##################
# Transformation #
##################
df = pd.read_csv(f'{output_dir}{log_file}_structured.csv')

# data preprocess
df["Label"] = df["Label"].apply(lambda x: int(x != "-"))

df['datetime'] = pd.to_datetime(df["Date"] + " " + df['Time'], format='%Y-%m-%d %H:%M:%S')
df['timestamp'] = df["datetime"].values.astype(np.int64) // 10 ** 9
df['deltaT'] = df['datetime'].diff() / np.timedelta64(1, 's')
df['deltaT'].fillna(0)

# 20M rows
raw_data = df[["timestamp", "Label", "EventId", "deltaT"]]
para={"window_size": float(window_size)*60, "step_size": float(step_size) * 60}

log_size = raw_data.shape[0]
label_data, time_data = raw_data.iloc[:, 1], raw_data.iloc[:, 0]
logkey_data, deltaT_data = raw_data.iloc[:, 2], raw_data.iloc[:, 3]
new_data = []
start_end_index_pair = set()

start_time = time_data[0]
end_time = start_time + para["window_size"]
start_index = 0
end_index = 0

# get the first start, end index, end time
for cur_time in time_data:
    if cur_time < end_time:
        end_index += 1
    else:
        break

start_end_index_pair.add(tuple([start_index, end_index]))

# move the start and end index until next sliding window
num_session = 1
while end_index < log_size:
    start_time = start_time + para['step_size']
    end_time = start_time + para["window_size"]
    for i in range(start_index, log_size):
        if time_data[i] < start_time:
            i += 1
        else:
            break
    for j in range(end_index, log_size):
        if time_data[j] < end_time:
            j += 1
        else:
            break
    start_index = i
    end_index = j
    # when start_index == end_index, there is no value in the window
    if start_index != end_index:
        start_end_index_pair.add(tuple([start_index, end_index]))
    ####
    num_session += 1
    if num_session % 1000 == 0:
        print("process {} time window".format(num_session), end='\r')

for (start_index, end_index) in start_end_index_pair:
    dt = deltaT_data[start_index: end_index].values
    dt[0] = 0
    new_data.append([
        time_data[start_index: end_index].values,
        label_data[start_index: end_index].values,
        logkey_data[start_index: end_index].values,
        dt,
        max(label_data[start_index:end_index])
    ])

assert len(start_end_index_pair) == len(new_data)
print('there are %d instances (sliding windows) in this dataset\n' % len(start_end_index_pair))
column_name = ['timestamp', 'LabelSeq', 'EventId', 'deltaT', 'Label']
deeplog_df = pd.DataFrame(new_data, columns=column_name)

df_normal = deeplog_df[deeplog_df["Label"] == 0]
df_abnormal = deeplog_df[deeplog_df["Label"] == 1]

df_normal = df_normal.sample(frac=1, random_state=12).reset_index(drop=True) #shuffle
normal_len = len(df_normal)
train_len = int(train_ratio) if train_ratio >= 1 else int(normal_len * train_ratio)

train = df_normal[:train_len]
test_normal = df_normal[train_len:]

def deeplog_file_generator_full(filename, df, features):
    with open(filename, 'w') as f:
        for _, row in df.iterrows():
            for fname in features:
                val = row[fname]
                f.write(','.join([str(v) for v in val]) + ' ')
            f.write('\n')

deeplog_file_generator_full(os.path.join(output_dir, 'train_with_time'), train, ['timestamp', 'LabelSeq', 'EventId', 'deltaT'])

print("save df with column {}".format(column_name))
print("training size {}".format(train_len))
print("test normal size {}".format(normal_len - train_len))
print('test abnormal size {}'.format(len(df_abnormal)))




##########################
# resolving the unique event id shows in training and validation
import sys
sys.path.append('../')

import os
import pandas as pd
import numpy as np
from logparser import Spell, Drain
from tqdm import tqdm
from logdeep.dataset.session import sliding_window

tqdm.pandas()
pd.options.mode.chained_assignment = None  # default='warn'

data_dir = os.path.expanduser("../dataset/tbird/")
output_dir = "../output/tbird/sliding_window_60_60_1.0/"
raw_log_file = "Thunderbird.log"
sample_log_file = "Thunderbird_20M.log"
sample_window_size = 2*10**7
sample_step_size = 10**4
window_name = ''
log_file = sample_log_file

parser_type = 'drain'
#mins
window_size = 1
step_size = 1
# train_ratio = 1.0

df = pd.read_csv(f'{output_dir}{log_file}_structured.csv')

# data preprocess
df["Label"] = df["Label"].apply(lambda x: int(x != "-"))

df['datetime'] = pd.to_datetime(df["Date"] + " " + df['Time'], format='%Y-%m-%d %H:%M:%S')
df['timestamp'] = df["datetime"].values.astype(np.int64) // 10 ** 9
df['deltaT'] = df['datetime'].diff() / np.timedelta64(1, 's')
df['deltaT'].fillna(0)

deeplog_df = sliding_window(
    df[["timestamp", "Label", "EventId", "deltaT"]],
    para={"window_size": float(window_size)*60, "step_size": float(step_size) * 60})
output_dir += window_name

df_normal = deeplog_df[deeplog_df["Label"] == 0]
df_abnormal = deeplog_df[deeplog_df["Label"] == 1]

