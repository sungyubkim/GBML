import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import higher

from gbml.gbml import GBML
from utils import get_accuracy, apply_grad, mix_grad, grad_to_cos, loss_to_ent

class MAML(GBML):

    def __init__(self, args):
        super().__init__(args)
        self._init_net()
        self._init_opt()
        return None

    @torch.enable_grad()
    def inner_loop(self, fmodel, diffopt, train_input, train_target):

        train_logit = fmodel(train_input)
        inner_loss = F.cross_entropy(train_logit, train_target)
        diffopt.step(inner_loss)

        return None

    def outer_loop(self, batch, is_train):

        self.network.zero_grad()
        
        train_inputs, train_targets, test_inputs, test_targets = self.unpack_batch(batch)

        loss_log = 0
        acc_log = 0
        grad_list = []
        loss_list = []

        for (train_input, train_target, test_input, test_target) in zip(train_inputs, train_targets, test_inputs, test_targets):
            with higher.innerloop_ctx(self.network, self.inner_optimizer, track_higher_grads=is_train) as (fmodel, diffopt):

                for step in range(self.args.n_inner):
                    self.inner_loop(fmodel, diffopt, train_input, train_target)

                test_logit = fmodel(test_input)
                outer_loss = F.cross_entropy(test_logit, test_target)
                loss_log += outer_loss.item()/self.batch_size

                with torch.no_grad():
                    acc_log += get_accuracy(test_logit, test_target).item()/self.batch_size
            
                if is_train:
                    outer_grad = torch.autograd.grad(outer_loss, fmodel.parameters(time=0))
                    grad_list.append(outer_grad)
                    loss_list.append(outer_loss.item())

        if is_train:
            weight = torch.ones(len(grad_list))
            weight = weight / torch.sum(weight)
            grad = mix_grad(grad_list, weight)
            grad_log = apply_grad(self.network, grad)
            self.outer_optimizer.step()
            
            return loss_log, acc_log, grad_log
        else:
            return loss_log, acc_log