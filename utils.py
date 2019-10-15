import os
import random
import functools
from collections import OrderedDict

import numpy as np
import torch

def get_accuracy(logits, targets):
    """Compute the accuracy (after adaptation) of MAML on the test/query points
    Parameters
    ----------
    logits : `torch.FloatTensor` instance
        Outputs/logits of the model on the query points. This tensor has shape
        `(num_examples, num_classes)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has 
        shape `(num_examples,)`.
    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points
    """
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())

def apply_grad(model, grad):
    '''
    assign gradient to model(nn.Module) instance. return the norm of gradient
    '''
    grad_norm = 0
    for p, g in zip(model.parameters(), grad):
        if p.grad is None:
            p.grad = g
        else:
            p.grad += g
        grad_norm += torch.sum(g**2)
    grad_norm = grad_norm ** (1/2)
    return grad_norm.item()

def mix_grad(grad_list, weight_list):
    '''
    calc weighted average of gradient
    '''
    mixed_grad = []
    for g_list in zip(*grad_list):
        g_list = torch.stack([weight_list[i] * g_list[i] for i in range(len(weight_list))])
        mixed_grad.append(torch.sum(g_list, dim=0))
    return mixed_grad

def grad_to_cos(grad_list):
    '''
    generate cosine similarity from list of gradient
    '''
    cos = 0.
    for g_list in zip(*grad_list):
        g_list = torch.stack(g_list)
        g_list = g_list.reshape(g_list.shape[0], -1) # (n, p)
        g_sum = torch.sum(g_list,dim=0) # (p)
        cos += torch.sum(g_list * g_sum.unsqueeze(0), dim=1) # (n)
    cos = cos/torch.sum(cos)
    return cos

def loss_to_ent(loss_list, lamb=1.0, beta=1.0):
    '''
    generate entropy weight from list of loss (uncertainty in loss function)
    '''
    loss_list = np.array(loss_list)
    ent = 1./(lamb + beta * loss_list)
    return ent

def set_seed(seed):
    # for reproducibility. 
    # note that pytorch is not completely reproducible 
    # https://pytorch.org/docs/stable/notes/randomness.html  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.initial_seed() # dataloader multi processing 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None

def set_gpu(x):
    x = [str(e) for e in x]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(x)
    print('using gpu:', ','.join(x))

def check_dir(args):
    # save path
    path = os.path.join(args.result_path, args.alg)
    if not os.path.exists(path):
        os.makedirs(path)
    return None

# https://github.com/sehkmg/tsvprint/blob/master/utils.py
def dict2tsv(res, file_name):
    if not os.path.exists(file_name):
        with open(file_name, 'a') as f:
            f.write('\t'.join(list(res.keys())))
            f.write('\n')

    with open(file_name, 'a') as f:
        f.write('\t'.join([str(r) for r in list(res.values())]))
        f.write('\n')

class BestTracker:
    '''Decorator for train function.
       Get ordered dict result (res),
       track best accuracy (self.best_acc) & best epoch (self.best_epoch) and
       append them to ordered dict result (res).
       Also, save the best result to file (best.txt).
       Return ordered dict result (res).'''
    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func
        self.best_epoch = 0
        self.best_valid_acc = 0
        self.best_test_acc = 0

    def __call__(self, *args, **kwargs):
        res = self.func(*args, **kwargs)

        if res['valid_acc'] > self.best_valid_acc:
            self.best_epoch = res['epoch']
            self.best_valid_acc = res['valid_acc']
            self.best_test_acc = res['test_acc']
            is_best = True
        else:
            is_best = False

        res['best_epoch'] = self.best_epoch
        res['best_valid_acc'] = self.best_valid_acc
        res['best_test_acc'] = self.best_test_acc

        return res, is_best