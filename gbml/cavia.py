import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import higher

from gbml.gbml import GBML
from utils import get_accuracy, apply_grad, mix_grad, grad_to_cos, loss_to_ent

class CAVIA(GBML):

    def __init__(self, args):
        super().__init__(args)
        self._init_net()
        self.context_dim = 200
        self.network.film = nn.Linear(self.context_dim, self.args.hidden_channels * 2).cuda()
        self._init_opt()
        return None

    def cavia_forward(self, x, context):
        '''
        context-dependent forward function for CAVIA.
        '''
        x = self.network.encoder(x) # (N, 64, 5, 5)
        transform = self.network.film(context) # (1, 64 * 2)
        scale, shift = transform[:self.args.hidden_channels], transform[self.args.hidden_channels:]
        scale, shift = scale.reshape(1, self.args.hidden_channels, 1, 1), shift.reshape(1, self.args.hidden_channels, 1, 1)
        x = scale * x + shift # (N, 64, 5, 5)
        x = self.network.decoder(x.reshape(x.shape[0], -1)) # (N, 5)

        return x

    @torch.enable_grad()
    def inner_loop(self, context, train_input, train_target, is_train):

        train_logit = self.cavia_forward(train_input, context)
        inner_loss = F.cross_entropy(train_logit, train_target)
        grad = torch.autograd.grad(inner_loss, context, create_graph=is_train)[0]
        context = context - self.args.inner_lr * grad

        return context

    def outer_loop(self, batch, is_train):

        self.network.zero_grad()
        train_inputs, train_targets = batch['train']
        train_inputs = train_inputs.cuda()
        train_targets = train_targets.cuda()

        test_inputs, test_targets = batch['test']
        test_inputs = test_inputs.cuda()
        test_targets = test_targets.cuda()

        loss_log = 0
        acc_log = 0
        grad_list = []
        loss_list = []

        for (train_input, train_target, test_input, test_target) in zip(train_inputs, train_targets, test_inputs, test_targets):
            context = torch.zeros(self.context_dim).cuda().requires_grad_()

            for step in range(self.args.n_inner):
                context = self.inner_loop(context, train_input, train_target, is_train)

            test_logit = self.cavia_forward(test_input, context)
            outer_loss = F.cross_entropy(test_logit, test_target)
            loss_log += outer_loss.item()/self.batch_size

            with torch.no_grad():
                acc_log += get_accuracy(test_logit, test_target).item()/self.batch_size
        
            if is_train:
                outer_grad = torch.autograd.grad(outer_loss, self.network.parameters())
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