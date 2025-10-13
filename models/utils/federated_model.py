import numpy as np
import torch.nn as nn
import torch
import torchvision
from argparse import Namespace
from utils.conf import get_device
from utils.conf import checkpoint_path
from utils.util import create_if_not_exists, generate_federated_mask
import os
import copy
import torch_geometric
import math

import pickle  
import os




class FederatedModel(nn.Module):
    """
    Federated learning model.
    """
    NAME = None
    N_CLASS = None

    def __init__(self, nets_list: list,
                 args: Namespace, transform: torchvision.transforms) -> None:
        super(FederatedModel, self).__init__()
        self.nets_list = nets_list
        self.args = args
        self.transform = transform

        
        self.random_state = np.random.RandomState()
        self.online_num = np.ceil(self.args.parti_num * self.args.online_ratio).item()
        self.online_num = int(self.online_num)

        self.global_net = None
        self.global_net_pre = None
        self.device = get_device(device_id=self.args.device_id)

        self.local_epoch = args.local_epoch
        self.local_lr = args.local_lr
        self.trainloaders = None
        self.testlodaers = None

        self.epoch_index = 0 

        folder_name = self.args.model + ('pFedIA' if getattr(self.args, 'proposed', False) else '')
        self.checkpoint_path = os.path.join(checkpoint_path(), self.args.dataset, self.args.structure, folder_name)
        create_if_not_exists(self.checkpoint_path)
        self.net_to_device()

        self.local_gradients = {}

        self.layer_filter = None

    def net_to_device(self):
        for net in self.nets_list:
            net.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_scheduler(self):
        return

    def ini(self):
        pass

    def col_update(self, communication_idx, publoader):
        pass

    def loc_update(self, priloader_list):
        pass

    def load_pretrained_nets(self):
        if self.load:
            for j in range(self.args.parti_num):
                pretrain_path = os.path.join(self.checkpoint_path, 'pretrain')
                save_path = os.path.join(pretrain_path, str(j) + '.ckpt')
                self.nets_list[j].load_state_dict(torch.load(save_path, self.device))
        else:
            pass

    def copy_nets2_prevnets(self):
        nets_list = self.nets_list
        prev_nets_list = self.prev_nets_list
        for net_id, net in enumerate(nets_list):
            net_para = net.state_dict()
            prev_net = prev_nets_list[net_id]
            prev_net.load_state_dict(net_para)

    def aggregate_nets(self, freq=None):
        save_dir = os.path.join(self.checkpoint_path, "gradients")
        os.makedirs(save_dir, exist_ok=True)
        def save_data(data, filename):
            with open(os.path.join(save_dir, filename), 'wb') as f:
                pickle.dump(data, f)
        if self.args.proposed:
            online_clients = self.online_clients
            device = self.device
            global_net = self.global_net

            
            if not hasattr(self, 'avg_grad_history'):
                self.avg_grad_history = {
                    cid: {name: None for name in global_net.state_dict()}
                    for cid in range(self.args.parti_num)
                }
                self.grad_round_counts = {cid: 0 for cid in range(self.args.parti_num)}

            current_avg_grads = {}
            for cid in online_clients:
                current_avg_grads[cid] = {}
                
                n = self.grad_round_counts[cid]
                self.grad_round_counts[cid] += 1
                new_n = self.grad_round_counts[cid]
                for name, grad in self.local_gradients[cid].items():
                    if self.avg_grad_history[cid][name] is None:
                        
                        self.avg_grad_history[cid][name] = grad.clone()
                    else:
                        
                        self.avg_grad_history[cid][name] = (self.avg_grad_history[cid][name] * n + grad.clone()) / new_n
                    current_avg_grads[cid][name] = self.avg_grad_history[cid][name]

            def generate_mask_from_avg_grad(avg_grad_dict, mask_ratio):
                """Generate a federated mask based on top percentage importance scores."""

                global_importance = {}
                for name in avg_grad_dict[list(avg_grad_dict.keys())[0]]:
                    all_grads = [torch.abs(avg_grad_dict[cid][name]) for cid in avg_grad_dict]
                    global_importance[name] = torch.mean(torch.stack(all_grads), dim=0)

                total_elements = sum(v.numel() for v in global_importance.values())
                if total_elements == 0 or mask_ratio <= 0:
                    return {name: torch.zeros_like(importance) for name, importance in global_importance.items()}

                topk = max(1, int(total_elements * mask_ratio))
                topk = min(topk, total_elements)

                all_values = torch.cat([v.flatten() for v in global_importance.values()])
                top_indices = torch.topk(all_values, k=topk, largest=True).indices

                flat_mask = torch.zeros_like(all_values, dtype=torch.float32)
                flat_mask[top_indices] = 1.0

                mask = {}
                start_idx = 0
                for name, importance in global_importance.items():
                    num_elements = importance.numel()
                    mask[name] = flat_mask[start_idx: start_idx + num_elements].reshape(importance.shape)
                    start_idx += num_elements

                return mask
            mask = generate_mask_from_avg_grad(current_avg_grads, self.args.mask_ratio)

            
            model_deltas = {
                cid: {
                    name: -self.local_lr * current_avg_grads[cid][name]  
                    for name in global_net.state_dict()
                }
                for cid in online_clients
            }

            
            
            if not hasattr(self, 'momentum_weights'):
                self.momentum_weights = torch.ones(len(online_clients)).to(device) / len(online_clients)

            
            distances = []
            for cid in online_clients:
                delta_norm = 0.0
                for name in model_deltas[cid]:
                    masked_delta = mask[name] * model_deltas[cid][name]
                    delta_norm += torch.norm(masked_delta, p=2).item() ** 2
                distances.append(math.sqrt(delta_norm))

            
            normalized_dist = torch.softmax(torch.tensor(distances).to(device), dim=0)
            self.momentum_weights = (
                    self.args.delta_beta * self.momentum_weights +
                    (1 - self.args.delta_beta) * normalized_dist
            )
            client_weights = self.momentum_weights / self.momentum_weights.sum()

            
            global_gradients = {name: torch.zeros_like(param).to(device)
                                for name, param in global_net.named_parameters()}

            for idx, cid in enumerate(online_clients):
                for name in global_gradients:
                    
                    weight_tensor = client_weights[idx]
                    if model_deltas[cid][name].dim() > 0:
                        weight_tensor = weight_tensor.view([1] * model_deltas[cid][name].dim())

                    
                    masked_grad = (-model_deltas[cid][name] / self.local_lr) * mask[name]
                    global_gradients[name] += weight_tensor * masked_grad

            
            for name, param in global_net.named_parameters():
                param.data -= global_gradients[name]  

            
            for net in self.nets_list:
                for name, param in net.named_parameters():
                    param.data += global_gradients[name]

        else:
            
            online_clients = self.online_clients
            device = self.device
            global_gradients = {name: torch.zeros_like(param).to(device)
                                for name, param in self.global_net.named_parameters()}

            
            if self.args.averaging == 'weight':
                online_clients_len = [len(self.trainloaders[i]) for i in online_clients]
                total_len = sum(online_clients_len)
                freq = [l / total_len for l in online_clients_len]
            else:
                freq = [1 / len(online_clients)] * len(online_clients)

            
            for idx, client_id in enumerate(online_clients):
                client_grads = self.local_gradients[client_id]
                for name in global_gradients:
                    global_gradients[name] += client_grads[name] * freq[idx]

            
            for name, param in self.global_net.named_parameters():
                param.data -= self.local_lr * global_gradients[name]

            
            
            for net in self.nets_list:
                for name, param in net.named_parameters():
                    param.data -= self.local_lr * global_gradients[name]



