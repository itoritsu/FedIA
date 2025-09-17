import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm
import copy
import random
from typing import List

from utils.args import *
from models.utils.federated_model import FederatedModel
from utils.util import diff_loss


class FedAvGg(FederatedModel):
    """FedAvg for graph backbones with layer‑wise Activation & CKA visualisation."""

    NAME = 'fedavggraph'
    COMPATIBILITY = ['homogeneity']

    
    def __init__(self, nets_list, args, transform):
        super().__init__(nets_list, args, transform)
        self.local_gradients = {}

    
    def ini(self):
        """Synchronise all local nets with the first net."""
        self.global_net = copy.deepcopy(self.nets_list[0])
        g_state = self.global_net.state_dict()
        for net in self.nets_list:
            net.load_state_dict(g_state)

    
    
    def loc_update(self, loaders):
        """One federated communication round."""
        clients = self.random_state.choice(range(self.args.parti_num),
                                           self.online_num, replace=False)
        self.online_clients = list(clients)
        self.global_net_pre = copy.deepcopy(self.global_net).to(self.device)

        
        for cid in clients:
            self._train_net(cid, self.nets_list[cid], loaders[cid])

        
        self.aggregate_nets(None)

        return

    
    def _train_net(self, cid, net, loader):
        """Local training for one client."""
        net = net.to(self.device)
        net.train()
        opt = optim.SGD(net.parameters(), lr=self.local_lr,
                        momentum=0.9, weight_decay=1e-5)
        crit = F.nll_loss
        grads = {n: torch.zeros_like(p).to(self.device) for n, p in net.named_parameters()}
        batches = 0

        for ep in range(self.local_epoch):
            if isinstance(loader, DataLoader):
                for raw in loader:
                    data = raw if hasattr(raw, 'x') else raw[0]
                    data = data.to(self.device)
                    opt.zero_grad()
                    out = net(data)
                    loss = crit(out[data.train_mask], data.y[data.train_mask])
                    loss.backward()
                    utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                    for n, p in net.named_parameters():
                        if p.grad is not None:
                            grads[n] += p.grad.clone()
                    batches += 1
            else:
                data = loader if hasattr(loader, 'x') else loader[0]
                data = data.to(self.device)
                opt.zero_grad()
                out = net(data)
                loss = crit(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                for n, p in net.named_parameters():
                    if p.grad is not None:
                        grads[n] += p.grad.clone()
                batches += 1

        for n in grads:
            grads[n] /= max(1, batches)
        self.local_gradients[cid] = grads

