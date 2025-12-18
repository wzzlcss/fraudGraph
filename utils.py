import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score

def compute_batch_auc(y_list, pred_list):
    all_pred = torch.cat(pred_list, dim=0)
    all_y = torch.cat(y_list, dim=0)
    all_y, all_pred = all_y.cpu().numpy(), all_pred.cpu().numpy()
    auc = roc_auc_score(all_y, all_pred)
    return auc

def compute_batch_pr_auc(y_list, pred_list):
    all_pred = torch.cat(pred_list, dim=0)
    all_y = torch.cat(y_list, dim=0)
    all_y, all_pred = all_y.cpu().numpy(), all_pred.cpu().numpy()
    pr_auc = average_precision_score(all_y, all_pred)
    return pr_auc

# # find common positive among training sequence
# common_positive = find_all_edges(train_eventSeq)
# common_positive = remove_self_loop(common_positive)
# common_negative = find_all_negative_edges(
#     common_positive, n, exclude_diag=True)
# common_negative = remove_self_loop(common_negative)

# total_positive = 0.0
# total_size = (n*n) * len(valid_eventSeq)
# for seq in valid_eventSeq:
#     num_positive = int(len(seq)/2) - 1
#     total_positive += num_positive

# inductive: multiple graphs
# input to the model: edge_index_unmasked; n, feat; edge_index_masked_for_loss 
# n_train_sample = len(train_dataloader)

# all_edges_n = all_edges(n)
# all_edges_n = torch.tensor(all_edges_n).to(device).to(dtype=torch.long)

# edge_index_empty = torch.empty(2, 0, dtype=torch.long)
# self_loop_only, _ = add_self_loops(edge_index_empty, num_nodes = n)
# self_loop_only = self_loop_only.to(device).to(dtype=torch.long)

def find_nontrivial_negative(input_edge, output_edge, n, common_positive):
    # new negative compared with t-1
    new_negative = find_new_negative(input_edge, output_edge, n)
    # curr negative that is in common positive
    curr_negative_in_common_positive, _ = curr_negative_vs_common_positive(
        output_edge, common_positive, n)
    # negative that is relevant to the current graph
    # relevant_negative = neg_edges_relevant(output_edge, n)
    return new_negative, curr_negative_in_common_positive #, relevant_negative

def sample_edges(full_set, target_size=500):
    if full_set.shape[1] <= target_size:
        return full_set
    idx = torch.randint(
    full_set.shape[1], (target_size,), device=full_set.device)
    return full_set[:, idx]

def compute_batch_false_positive(pred_for_negative):
    all_pred = torch.cat(pred_for_negative, dim=0).squeeze()
    FP = (all_pred == 1).sum().item()
    return FP / len(all_pred)

def compute_batch_false_negative(pred_for_positive):
    all_pred = torch.cat(pred_for_positive, dim=0).squeeze()
    FN = (all_pred == 0).sum().item()
    return FN / len(all_pred)

def evaluate_negative(z, negative):
    negative = torch.tensor(negative).to(device).to(dtype=torch.long)
    # negative = sample_edges(
    #     negative, target_size=500)
    return model.evaluate_certain(z, negative)

def evaluate_positive(z, positive):
    positive = torch.tensor(positive).to(device).to(dtype=torch.long)
    return model.evaluate_certain(z, positive)