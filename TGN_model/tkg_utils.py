
from itertools import groupby
from operator import itemgetter
from collections import defaultdict
import sys
import argparse
import numpy as np

def get_args_cen():
    """ Get the arguments for "CEN" model"""
    parser = argparse.ArgumentParser(description='CEN')
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch-size")
    parser.add_argument("-d", "--dataset", type=str, default='tkgl-yago',
                        help="dataset to use")
    parser.add_argument("--test", type=int, default=0,
                        help="1: formal test 2: continual test")
    parser.add_argument("--validtest",  default=False,
                        help="load stat from dir and directly valid and test")
    parser.add_argument("--test-only", type=bool, default=False,
                        help="do we want to compute valid mrr or only test")
    parser.add_argument("--run-statistic", action='store_true', default=False,
                        help="statistic the result")

    parser.add_argument("--relation-evaluation", action='store_true', default=False,
                        help="save model accordding to the relation evalution")
    parser.add_argument("--log-per-rel", action='store_true', default=False,
                        help="log mrr per relation in json")

    
    # configuration for encoder RGCN stat
    parser.add_argument("--weight", type=float, default=1,
                        help="weight of static constraint")
    parser.add_argument("--task-weight", type=float, default=1,
                        help="weight of entity prediction task")
    parser.add_argument("--kl-weight", type=float, default=0.7,
                        help="weight of entity prediction task")
   
    parser.add_argument("--encoder", type=str, default="uvrgcn",
                        help="method of encoder")

    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False,
                        help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--n-hidden", type=int, default=200,
                        help="number of hidden units")
    parser.add_argument("--opn", type=str, default="sub",
                        help="opn of compgcn")

    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100,
                        help="number of basis vector for compgcn")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", action='store_true', default=False,
                        help="add relation prediction loss")
    parser.add_argument("--entity-prediction", action='store_true', default=True,
                        help="add entity prediction loss")


    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=30,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--ft_epochs", type=int, default=30,
                        help="number of minimum fine-tuning epoch")
    parser.add_argument("--ft_lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--norm_weight", type=float, default=1,
                        help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")

    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=1,
                        help="perform evaluation every n epochs")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="convtranse",
                        help="method of decoder")
    parser.add_argument("--input-dropout", type=float, default=0.2,
                        help="input dropout for decoder ")
    parser.add_argument("--hidden-dropout", type=float, default=0.2,
                        help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2,
                        help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train-history-len", type=int, default=3,
                        help="history length")
    parser.add_argument("--test-history-len", type=int, default=10,
                        help="history length for test")
    parser.add_argument("--test-history-len-2", type=int, default=2,
                        help="history length for test")
    parser.add_argument("--start-history-len", type=int, default=3,
                    help="start history length")
    parser.add_argument("--dilate-len", type=int, default=1,
                        help="dilate history graph")

    # configuration for optimal parameters
    parser.add_argument("--grid-search", action='store_true', default=False,
                        help="perform grid search for best configuration")
    parser.add_argument("-tune", "--tune", type=str, default="n_hidden,n_layers,dropout,n_bases",
                        help="stat to use")
    parser.add_argument("--num-k", type=int, default=500,
                        help="number of triples generated")
    parser.add_argument('--seed', type=int, help='Random seed', default=1)
    parser.add_argument('--run-nr', type=int, help='Run Number', default=1)

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv 

def get_args_regcn():
    """Parses the arguments for REGCN model"""
    parser = argparse.ArgumentParser(description='REGCN')

    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch-size")
    parser.add_argument("-d", "--dataset", type=str, default='tkgl-yago',
                        help="dataset to use")
    parser.add_argument("--test", default=False,
                        help="load stat from dir and directly test")
    parser.add_argument("--run-analysis", action='store_true', default=False,
                        help="print log info")
    parser.add_argument("--run-statistic", action='store_true', default=False,
                        help="statistic the result")
    parser.add_argument("--multi-step", action='store_true', default=False,
                        help="do multi-steps inference without ground truth")
    parser.add_argument("--topk", type=int, default=10,
                        help="choose top k entities as results when do multi-steps without ground truth")
    parser.add_argument("--add-static-graph",  action='store_true', default=False,
                        help="use the info of static graph")
    parser.add_argument("--add-rel-word", action='store_true', default=False,
                        help="use words in relaitons")
    parser.add_argument("--relation-evaluation", action='store_true', default=False,
                        help="save model accordding to the relation evalution")

    # configuration for encoder RGCN stat

    parser.add_argument("--weight", type=float, default=0.5,
                        help="weight of static constraint")
    parser.add_argument("--task-weight", type=float, default=0.7,
                        help="weight of entity prediction task")
    parser.add_argument("--discount", type=float, default=1.0,
                        help="discount of weight of static constraint")
    parser.add_argument("--angle", type=int, default=10,
                        help="evolution speed")

    parser.add_argument("--encoder", type=str, default="uvrgcn",
                        help="method of encoder")
    parser.add_argument("--aggregation", type=str, default="none",
                        help="method of aggregation")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False,
                        help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--n-hidden", type=int, default=200,
                        help="number of hidden units")
    parser.add_argument("--opn", type=str, default="sub",
                        help="opn of compgcn")

    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100,
                        help="number of basis vector for compgcn")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", action='store_true', default=False,
                        help="add relation prediction loss")
    parser.add_argument("--entity-prediction", action='store_true', default=True,
                        help="add entity prediction loss")
    parser.add_argument("--split_by_relation", action='store_true', default=False,
                        help="do relation prediction")

    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="number of minimum training epochs on each time step") #100
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")

    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=1,
                        help="perform evaluation every n epochs")
    parser.add_argument("--log-per-rel", action='store_true', default=False,
                        help="log mrr per relation in json")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="convtranse",
                        help="method of decoder")
    parser.add_argument("--input-dropout", type=float, default=0.2,
                        help="input dropout for decoder ")
    parser.add_argument("--hidden-dropout", type=float, default=0.2,
                        help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2,
                        help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train-history-len", type=int, default=3,
                        help="history length")
    parser.add_argument("--test-history-len", type=int, default=3,
                        help="history length for test")
    parser.add_argument("--dilate-len", type=int, default=1,
                        help="dilate history graph")

    # configuration for optimal parameters
    parser.add_argument("--grid-search", action='store_true', default=False,
                        help="perform grid search for best configuration")
    parser.add_argument("-tune", "--tune", type=str, default="n_hidden,n_layers,dropout,n_bases",
                        help="stat to use")
    parser.add_argument("--num-k", type=int, default=500,
                        help="number of triples generated")
    parser.add_argument('--seed', type=int, help='Random seed', default=1)
    parser.add_argument('--run-nr', type=int, help='Run Number', default=1)
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv 


def compute_min_distance(unique_sorted_timestamps):
    """ compute the minimum distance between timestamps, where the timestamps are in a sorted list
    """
    min_distance = np.inf
    for i in range(1, len(unique_sorted_timestamps)):
        min_distance = min(min_distance, unique_sorted_timestamps[i] - unique_sorted_timestamps[i-1])
    return min_distance

def compute_maxminmean_distances(unique_sorted_timestamps):
    """ compute the maximum, minimum and mean distances between timestamps, where the timestamps are in a sorted list"""
    differences = []
    
    # Iterate over the list and compute the differences between successive elements
    for i in range(len(unique_sorted_timestamps) - 1):
        diff = unique_sorted_timestamps[i+1] - unique_sorted_timestamps[i]
        differences.append(diff)
    
    # Calculate the mean of the differences
    mean_diff = sum(differences) / len(differences)
    
    return np.max(differences), np.min(differences), np.mean(differences)

def group_by(data: np.array, key_idx: int) -> dict:
    """
    group data in an np array to dict; where key is specified by key_idx. for example groups elements of array by relations
    :param data: [np.array] data to be grouped
    :param key_idx: [int] index for element of interest
    returns data_dict: dict with key: values of element at index key_idx, values: all elements in data that have that value
    """
    data_dict = {}
    data_sorted = sorted(data, key=itemgetter(key_idx))
    for key, group in groupby(data_sorted, key=itemgetter(key_idx)):
        data_dict[key] = np.array(list(group))
    return data_dict

def tkg_granularity_lookup(dataset_name, ts_distmean):
    """ lookup the granularity of the dataset, and return the corresponding granularity
    """
    if 'icews' in dataset_name or 'polecat' in dataset_name:
        return 86400
    elif 'wiki' in dataset_name or 'yago' in dataset_name:
        return 31536000
    else:
        return ts_distmean

    

def reformat_ts(timestamps, dataset_name='tkgl'):
    """ reformat timestamps s.t. they start with 0, and have stepsize 1.
    :param timestamps: np.array() with timestamps
    returns: np.array(ts_new)
    """
    all_ts = list(set(timestamps))
    all_ts.sort()
    ts_min = np.min(all_ts)
    if 'tkgl' in dataset_name:
        ts_distmax, ts_distmin, ts_distmean = compute_maxminmean_distances(all_ts)
        if ts_distmean != ts_distmin:
            ts_dist = tkg_granularity_lookup(dataset_name, ts_distmean)
            if ts_dist - ts_distmean > 0.1*ts_distmean:
                print('PROBLEM: the distances are somehwat off from the granularity of the dataset. using original mean distance')
                ts_dist = ts_distmean
        else:
            ts_dist = ts_distmean
    else:
        ts_dist = compute_min_distance(all_ts) # all_ts[1] - all_ts[0]

    ts_new = []
    timestamps2 = timestamps - ts_min
    ts_new = np.ceil(timestamps2/ts_dist).astype(int)

    return np.array(ts_new)

def get_original_ts(reformatted_ts, ts_dist, min_ts):
    """ get original timestamps from reformatted timestamps
    :param reformatted_ts: np.array() with reformatted timestamps
    returns: np.array(ts_new)
    """
    reformatted_ts = list(set(reformatted_ts))
    reformatted_ts.sort()
    ts_new = []
    for ts in reformatted_ts:
        ts_new.append((ts * ts_dist)+min_ts)
    return np.array(ts_new)


def create_basis_dict(data):
    """
    Create basis dictionary for the recurrency baseline model with rules of confidence 1
    data: concatenated train and vali data, INCLUDING INVERSE QUADRUPLES. we need it for the relation ids.
    """
    rels = list(set(data[:,1]))
    basis_dict = {}
    for rel in rels:
        basis_id_new = []
        rule_dict = {}
        rule_dict["head_rel"] = int(rel)
        rule_dict["body_rels"] = [int(rel)] #same body and head relation -> what happened before happens again
        rule_dict["conf"] = 1 #same confidence for every rule
        rule_new = rule_dict
        basis_id_new.append(rule_new)
        basis_dict[str(rel)] = basis_id_new
    return basis_dict


def get_inv_relation_id(num_rels):
    """
    Get inverse relation id.
    parameters:
        num_rels (int): number of relations
    returns:
        inv_relation_id (dict): mapping of relation to inverse relation
    """
    inv_relation_id = dict()
    for i in range(int(num_rels / 2)):
        inv_relation_id[i] = i + int(num_rels / 2)
    for i in range(int(num_rels / 2), num_rels):
        inv_relation_id[i] = i % int(num_rels / 2)
    return inv_relation_id


def create_scores_array(predictions_dict, num_nodes):
    """ 
    Create an array of scores from a dictionary of predictions.
    predictions_dict: a dictionary mapping indices to values
    num_nodes: the size of the array
    returns: an array of scores
    """
    # predictions_dict is a dictionary mapping indices to values
    # num_nodes is the size of the array

    # Convert keys and values of the predictions_dict into NumPy arrays
    keys_array = np.array(list(predictions_dict.keys()))
    values_array = np.array(list(predictions_dict.values()))

    # Create an array of zeros with the desired shape
    predictions = np.zeros(num_nodes)

    # Use advanced indexing to scatter values into predictions array
    predictions[keys_array.astype(int)] = values_array.astype(float)
    return predictions

