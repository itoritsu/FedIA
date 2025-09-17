import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel





def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning via FedOpt.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class FedOptg(FederatedModel):
    NAME = 'fedoptg'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedOptg, self).__init__(nets_list, args, transform)

        self.global_lr = 0.5  

        self.global_optimizer=None

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        self.global_optimizer = torch.optim.SGD(
            self.global_net.parameters(),
            lr=self.global_lr,
            momentum=0.9,
            weight_decay=0.0
        )
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def update_global(self):
        online_clients = self.online_clients
        device = self.device

        
        global_gradients = {
            name: torch.zeros_like(param).to(device)
            for name, param in self.global_net.named_parameters()
        }

        
        if self.args.averaging == 'weight':
            online_clients_len = [len(self.trainloaders[i]) for i in online_clients]
            total_len = sum(online_clients_len)
            freq = [cl_len / total_len for cl_len in online_clients_len]
        else:
            freq = [1.0 / len(online_clients)] * len(online_clients)

        
        for idx, client_id in enumerate(online_clients):
            client_grads = self.local_gradients[client_id]  
            for name in global_gradients:
                global_gradients[name] += client_grads[name] * freq[idx]

        
        global_optimizer_state = self.global_optimizer.state_dict()

        
        self.global_optimizer.zero_grad()
        with torch.no_grad():
            for name, param in self.global_net.named_parameters():
                param.grad = global_gradients[name].clone()

        
        self.global_optimizer.load_state_dict(global_optimizer_state)
        self.global_optimizer.step()

        
        for net in self.nets_list:
            net.load_state_dict(self.global_net.state_dict())

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

        
        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i])

        self.update_global()
        return None

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()
        train_loader = train_loader.to(self.device)
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9,weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.local_epoch))
        gradients = {name: torch.zeros_like(param).to(self.device) for name, param in net.named_parameters()}
        for _ in iterator:
            outputs = net(train_loader)
            loss = criterion(outputs[train_loader.train_mask], train_loader.y[train_loader.train_mask])
            optimizer.zero_grad()
            loss.backward()
            for name, param in net.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.clone()
            iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)
            optimizer.step()
        for name in gradients:
            gradients[name] /= self.local_epoch

        self.local_gradients[index] = gradients
