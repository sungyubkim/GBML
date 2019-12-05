import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import higher
import copy

from gbml.gbml import GBML
from utils import get_accuracy, apply_grad, mix_grad, grad_to_cos, loss_to_ent

class Neumann(GBML):

    def __init__(self, args):
        super().__init__(args)
        self._init_net()
        self.network_aux = type(self.network)(args).cuda()
        self.network_aux.load_state_dict(self.network.state_dict())
        self._init_opt()
        self.inner_optimizer = torch.optim.SGD(self.network_aux.parameters(), lr=self.args.inner_lr)
        self.n_series = 5
        return None

    @torch.enable_grad()
    def inner_loop(self, train_input, train_target):
        self.network_aux.zero_grad()
        train_logit = self.network_aux(train_input)
        inner_loss = F.cross_entropy(train_logit, train_target)
        inner_loss.backward()
        self.inner_optimizer.step()
        return None

    @torch.enable_grad()
    def neumann_approx(self, in_grad, outer_grad):
        in_grad = torch.nn.utils.parameters_to_vector(in_grad)
        outer_grad = torch.nn.utils.parameters_to_vector(outer_grad)
        x = outer_grad.clone().detach()
        for i in range(self.n_series):
            outer_grad = self.hv_prod(in_grad, outer_grad)
            x = x + outer_grad
        return self.vec_to_grad(x)
    
    def vec_to_grad(self, vec):
        pointer = 0
        res = []
        for param in self.network.parameters():
            num_param = param.numel()
            res.append(vec[pointer:pointer+num_param].view_as(param).data)
            pointer += num_param
        return res

    def hv_prod(self, in_grad, x):
        scalar = in_grad @ x.detach()
        hv = torch.autograd.grad(scalar, self.network_aux.parameters(), retain_graph=True)
        hv = torch.nn.utils.parameters_to_vector(hv)
        hv = (-1.*self.args.inner_lr) * hv # scale for regularization
        return hv.detach()

    def outer_loop(self, batch, is_train):
        
        train_inputs, train_targets, test_inputs, test_targets = self.unpack_batch(batch)

        loss_log = 0
        acc_log = 0
        grad_list = []
        loss_list = []

        for (train_input, train_target, test_input, test_target) in zip(train_inputs, train_targets, test_inputs, test_targets):

            self.network_aux.load_state_dict(self.network.state_dict())

            for step in range(self.args.n_inner):
                self.inner_loop(train_input, train_target)
            
            train_logit = self.network_aux(train_input)
            in_loss = F.cross_entropy(train_logit, train_target)

            test_logit = self.network_aux(test_input)
            outer_loss = F.cross_entropy(test_logit, test_target)
            loss_log += outer_loss.item()/self.batch_size

            with torch.no_grad():
                acc_log += get_accuracy(test_logit, test_target).item()/self.batch_size
        
            if is_train:
                in_grad = torch.autograd.grad(in_loss, self.network_aux.parameters(), create_graph=True)
                outer_grad = torch.autograd.grad(outer_loss, self.network_aux.parameters())
                implicit_grad = self.neumann_approx(in_grad, outer_grad)
                grad_list.append(implicit_grad)
                loss_list.append(outer_loss.item())

        if is_train:
            self.outer_optimizer.zero_grad()
            weight = torch.ones(len(grad_list))
            weight = weight / torch.sum(weight)
            grad = mix_grad(grad_list, weight)
            grad_log = apply_grad(self.network, grad)
            self.outer_optimizer.step()
            
            return loss_log, acc_log, grad_log
        else:
            return loss_log, acc_log