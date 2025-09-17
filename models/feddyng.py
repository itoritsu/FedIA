import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from models.utils.federated_model import FederatedModel
import torch
import torch.nn.functional as F
import numpy as np

class FedDyng(FederatedModel):
    NAME = 'feddyng'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedDyng, self).__init__(nets_list, args, transform)
        self.mu = args.mu
        self.local_gradients = {}
        self.local_counter = 0
        self.curr_commu = 0
        self.alpha = 0.01
        self.local_drift = {}

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i])

        self.aggregate_nets(None)
        return None

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        train_loader = train_loader.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = F.nll_loss
        iterator = tqdm(range(self.local_epoch))

        
        gradients = {name: torch.zeros_like(param).to(self.device)
                     for name, param in net.named_parameters()}

        
        global_weights = {name: param.data.clone()
                          for name, param in self.global_net.named_parameters()}

        
        prev_gradients = self.local_gradients.get(index, None)

        for _ in iterator:
            optimizer.zero_grad()
            out = net(train_loader)
            loss1 = criterion(out[train_loader.train_mask], train_loader.y[train_loader.train_mask])

            
            loss2 = 0
            if prev_gradients is not None:
                for name, param in net.named_parameters():
                    loss2 -= torch.sum(prev_gradients[name] * param)

            
            loss3 = 0
            for name, param in net.named_parameters():
                loss3 += (self.alpha / 2) * torch.norm(param - global_weights[name]) ** 2

            loss = loss1 + loss2 + loss3
            loss.backward()

            
            for name, param in net.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.clone()

            iterator.desc = f"Local Participant {index} loss = {loss.item():.3f}"
            optimizer.step()

        
        for name in gradients:
            gradients[name] /= self.local_epoch

        
        self.local_gradients[index] = gradients

        
        self.local_drift[index] = {
            name: net_param.data.clone() - global_weights[name]
            for name, net_param in net.named_parameters()
        }
