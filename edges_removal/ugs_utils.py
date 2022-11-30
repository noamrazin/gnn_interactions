
import torch


class L1MaskRegulazation():
    def __init__(self, s1):
        self.s1 = s1

    def __call__(self, model):
        return self.s1 * torch.norm(model.adj_mask_train, p=1)


def get_mask_distribution(model, if_numpy=True):

    adj_mask_tensor = model.adj_mask_train.flatten()
    nonzero = torch.abs(adj_mask_tensor) > 0
    adj_mask_tensor = adj_mask_tensor[nonzero] # 13264 - 2708


    if if_numpy:
        return adj_mask_tensor.detach().cpu().numpy(),
    else:
        return adj_mask_tensor.detach().cpu()

def get_each_mask(mask_weight_tensor, threshold):
    
    ones  = torch.ones_like(mask_weight_tensor)
    zeros = torch.zeros_like(mask_weight_tensor) 
    mask = torch.where(mask_weight_tensor.abs() > threshold, ones, zeros)
    return mask

def get_final_mask_epoch(model, adj_percent, total_edges_num):
    
    adj_mask = get_mask_distribution(model, if_numpy=False)
    #adj_mask.add_((2 * torch.rand(adj_mask.shape) - 1) * 1e-5)
    adj_total = adj_mask.shape[0]
    ### sort
    adj_y, adj_i = torch.sort(adj_mask.abs())
    #nn.Parameter(torch.ones(num_edges, dtype=torch.float32))## get threshold
    adj_thre_index = min(int(total_edges_num//2 * adj_percent), adj_y.shape[0]-1)
    adj_thre = adj_y[adj_thre_index]
    
    ori_adj_mask = model.adj_mask_train.detach().cpu()
    mask = get_each_mask(ori_adj_mask, adj_thre)

    return mask

def mask_init(num_edges, c=1e-5):
    mask = torch.ones(num_edges, dtype=torch.float32)
    rand1 = (2 * torch.rand(num_edges) - 1) * c
    rand1 = rand1 * mask
    return rand1 + mask

