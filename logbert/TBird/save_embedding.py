import sys
sys.path.append("../")
sys.path.append("../../")

import argparse
import torch

from bert_pytorch.dataset import WordVocab
from bert_pytorch import Predictor, Trainer
from bert_pytorch.dataset.utils import seed_everything

options = dict()
options['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
options["output_dir"] = "../output/tbird/"
options["model_dir"] = options["output_dir"] + "bert/"
options["train_vocab"] = options["output_dir"] + "train"
options["vocab_path"] = options["output_dir"] + "vocab.pkl"

options["model_path"] = options["model_dir"] + "best_bert.pth"

options["window_size"] = 128
options["adaptive_window"] = True
options["seq_len"] = 512
options["max_len"] = 512 # for position embedding
options["min_len"] = 10

options["mask_ratio"] = 0.5

options["vocab_size"] = 844

options["train_ratio"] = 1
options["valid_ratio"] = 0.1
options["test_ratio"] = 0.1


# features
options["is_logkey"] = True
options["is_time"] = False

options["hypersphere_loss"] = True
options["hypersphere_loss_test"] = False

options["scale"] = None # MinMaxScaler()
options["scale_path"] = options["model_dir"] + "scale.pkl"

# model
options["hidden"] = 256 # embedding size
options["layers"] = 4
options["attn_heads"] = 4

options["epochs"] = 200
options["n_epochs_stop"] = 10
options["batch_size"] = 32

options["corpus_lines"] = None
options["on_memory"] = True
options["num_workers"] = 5
options["lr"] = 1e-3
options["adam_beta1"] = 0.9
options["adam_beta2"] = 0.999
options["adam_weight_decay"] = 0.00
options["with_cuda"]= True
options["cuda_devices"] = None
options["log_freq"] = None

# predict
options["num_candidates"] = 15
options["gaussian_mean"] = 0
options["gaussian_std"] = 1

seed_everything(seed=1234)
print("device", options["device"])
print("features logkey:{} time: {}\n".format(options["is_logkey"], options["is_time"]))
print("mask ratio", options["mask_ratio"])

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

train_parser = subparsers.add_parser('train')
train_parser.set_defaults(mode='train')

predict_parser = subparsers.add_parser('predict')
predict_parser.set_defaults(mode='predict')
predict_parser.add_argument("-m", "--mean", type=float, default=0)
predict_parser.add_argument("-s", "--std", type=float, default=1)

vocab_parser = subparsers.add_parser('vocab')
vocab_parser.set_defaults(mode='vocab')
vocab_parser.add_argument("-s", "--vocab_size", type=int, default=None)
vocab_parser.add_argument("-e", "--encoding", type=str, default="utf-8")
vocab_parser.add_argument("-m", "--min_freq", type=int, default=1)

args = parser.parse_args()

model = torch.load(options["model_path"], weights_only=False)

embedding = model.bert.embedding.token.weight.data

# save embedding table
torch.save(embedding, options["model_dir"] + "embedding.pt")

vocab = WordVocab.load_vocab(options["vocab_path"])
predictor = Predictor(options)

model.to(predictor.device)
model.eval()

center_dict = torch.load(predictor.model_dir + "best_center.pt", weights_only=False)
predictor.center = center_dict["center"]
predictor.radius = center_dict["radius"]


# from training
from bert_pytorch.dataset.sample import generate_train_valid
from bert_pytorch.dataset import LogDataset
from torch.utils.data import DataLoader

trainer = Trainer(options)

logkey_train, logkey_valid, time_train, time_valid = generate_train_valid(
    trainer.output_path + "train", 
    window_size=trainer.window_size,
    adaptive_window=trainer.adaptive_window,
    valid_size=trainer.valid_ratio,
    sample_ratio=trainer.sample_ratio,
    scale=trainer.scale,
    scale_path=trainer.scale_path,
    seq_len=trainer.seq_len,
    min_len=trainer.min_len
)

# training data
train_dataset = LogDataset(
    logkey_train,
    time_train, 
    vocab, 
    seq_len=trainer.seq_len,
    corpus_lines=trainer.corpus_lines, 
    on_memory=trainer.on_memory, 
    mask_ratio=trainer.mask_ratio)

train_data_loader = DataLoader(
    train_dataset, batch_size=trainer.batch_size, num_workers=trainer.num_workers,
    collate_fn=train_dataset.collate_fn, drop_last=True)

valid_dataset = LogDataset(
    logkey_valid, 
    time_valid, 
    vocab, 
    seq_len=trainer.seq_len, 
    on_memory=trainer.on_memory, 
    mask_ratio=trainer.mask_ratio)

# validation data

data_loader = train_data_loader

# Setting the tqdm progress bar
total_length = len(data_loader)
# data_iter = tqdm.tqdm(enumerate(data_loader), total=totol_length)
data_iter = enumerate(data_loader)

i, data = next(iter(data_iter))

data = {key: value.to(trainer.device) for key, value in data.items()}

result = model.forward(data["bert_input"], data["time_input"])

