from tqdm import tqdm
import numpy as np

import torch
import argparse
from torch import Tensor

from graph_utils.load_seq_event import *
from graph_utils.edge_selection import *

from torch.utils.data import DataLoader

import random
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from TGN_model.edgebank_predictor import EdgeBankPredictor

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument('--bs', type=int, help='Batch size', default=200)
parser.add_argument('--k_value', type=int, help='k_value for computing ranking metrics', default=10)
parser.add_argument('--mem_mode', type=str, help='Memory mode', 
    default='unlimited', choices=['unlimited', 'fixed_time_window'])
# for mode unlimited, condition edges will all be memorized without considering time

args = parser.parse_args()
device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

MEMORY_MODE = args.mem_mode # `unlimited` or `fixed_time_window`
BATCH_SIZE = args.bs
K_VALUE = args.k_value
# TIME_WINDOW_RATIO = args.time_window_ratio
# DATA = args.data
# MODEL_NAME = 'EdgeBank'

train_dataloader, valid_dataloader, train_eventSeq, valid_eventSeq, feat, n = prepare_seq(
    data_name="amazon")

for step, batch in enumerate(valid_dataloader, start=1):
    input_edge, output_edge = batch
    # use input_edge to create src dst ts
    # Set EdgeBank with memory updater per testing sequence
    eb = EdgeBankPredictor(
        src=src, # source node id of the edges
        dst=dst, # destination node id of the edges
        ts=ts, # will not be used under 'unlimited' mode
        memory_mode=MEMORY_MODE,
        # time_window_ratio=TIME_WINDOW_RATIO
    )
    # create prediction query src and dst
    query_src = np.array([int(pos_src[idx]) for _ in range(len(neg_batch) + 1)])
    query_dst = np.concatenate([np.array([int(pos_dst[idx])]), neg_batch])
    y_pred = edgebank.predict_link(query_src, query_dst)
    # collect prediction to calculate overall performance