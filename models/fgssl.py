import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict, defaultdict
import numpy as np
from models.utils.federated_model import FederatedModel
import copy


class FGSSL(FederatedModel):
    NAME = 'fgssl'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super().__init__(nets_list, args, transform)
        self.global_protos = OrderedDict()  
        self.local_protos = {}  
        self._init_components(args)

    def _init_components(self, args):
        
        
        self.base_lr = args.local_lr
        self.prototype_momentum = getattr(args, 'proto_momentum', 0.9)
        self.prototype_temp = getattr(args, 'proto_temp', 0.07)
        self.proto_loss_weight = getattr(args, 'proto_loss_weight', 0.5)

        
        self.local_gradients = {}

        
        self.global_net = copy.deepcopy(self.nets_list[0])
        self._sync_global_to_local()

    def _sync_global_to_local(self):
        
        global_w = self.global_net.state_dict()
        for net in self.nets_list:
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        
        
        total_clients = list(range(self.args.parti_num))
        self.online_clients = self.random_state.choice(
            total_clients,
            self.online_num,
            replace=False
        ).tolist()

        
        self._init_gradient_storage()

        
        for cid in self.online_clients:
            self._train_client(cid, priloader_list[cid])

        
        self._aggregate()
        return None

    def _init_gradient_storage(self):
        
        self.local_gradients = {
            cid: None for cid in self.online_clients
        }

    def _train_client(self, cid, train_loader):
        
        try:
            net = self.nets_list[cid].to(self.device)
            train_loader = train_loader.to(self.device)

            
            optimizer = torch.optim.SGD(net.parameters(), lr=self.base_lr)
            for _ in range(self.local_epoch):
                loss = self._compute_loss(net, train_loader)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            
            self._collect_protos_and_grads(cid, net, train_loader)

        except Exception as e:
            print(f"Client {cid} training failed: {str(e)}")
            self._handle_failure(cid)

    def _compute_loss(self, net, train_loader):
        
        
        outputs = net(train_loader)
        features = net.features(train_loader)

        
        cls_loss = F.nll_loss(
            outputs[train_loader.train_mask],
            train_loader.y[train_loader.train_mask]
        )

        
        proto_loss = self._compute_proto_loss(features, outputs, train_loader)

        return cls_loss + self.proto_loss_weight * proto_loss

    def _compute_proto_loss(self, features, outputs, train_loader):
        
        
        with torch.no_grad():
            confidences = torch.softmax(outputs, dim=1).max(dim=1).values
            local_protos = self._generate_prototypes(features, train_loader.y, confidences)

        
        if not self.global_protos:
            return torch.tensor(0.0, device=self.device)

        
        common_labels = set(local_protos.keys()) & set(self.global_protos.keys())
        if not common_labels:
            return torch.tensor(0.0, device=self.device)

        
        local_vec = torch.stack([local_protos[l] for l in common_labels])
        global_vec = torch.stack([self.global_protos[l] for l in common_labels])
        return self._proto_contrastive_loss(local_vec, global_vec)

    def _generate_prototypes(self, features, labels, confidences):
        
        protos = OrderedDict()
        for label in torch.unique(labels):
            mask = labels == label
            if mask.sum() == 0:
                continue
            weighted = (features[mask] * confidences[mask].unsqueeze(1)).sum(0)
            protos[label.item()] = weighted / (confidences[mask].sum() + 1e-8)
        return protos

    def _proto_contrastive_loss(self, local_vec, global_vec):
        
        sim_matrix = F.cosine_similarity(
            local_vec.unsqueeze(1),
            global_vec.unsqueeze(0),
            dim=-1
        )
        logits = sim_matrix / self.prototype_temp
        targets = torch.arange(len(local_vec), device=self.device)
        return F.cross_entropy(logits, targets)

    def _collect_protos_and_grads(self, cid, net, train_loader):
        
        
        global_params = dict(self.global_net.named_parameters())

        
        client_grads = {
            name: torch.zeros_like(param)
            for name, param in global_params.items()
        }

        
        for name, param in net.named_parameters():
            if param.grad is not None:
                client_grads[name] = param.grad.clone()

        self.local_gradients[cid] = client_grads

    def _handle_failure(self, cid):
        
        global_params = dict(self.global_net.named_parameters())
        self.local_gradients[cid] = {
            name: torch.zeros_like(param)
            for name, param in global_params.items()
        }
        self.local_protos[cid] = {}

    def _aggregate_prototypes(self):
        
        agg_protos = defaultdict(list)

        
        for cid in self.online_clients:
            client_protos = self.local_protos.get(cid, {})
            for label, proto in client_protos.items():
                if proto is not None and not torch.isnan(proto).any():
                    agg_protos[label].append(proto.detach().cpu().numpy())

        
        for label, protos in agg_protos.items():
            if len(protos) == 0:
                continue

            try:
                mean_proto = np.nanmean(protos, axis=0)
                if np.isnan(mean_proto).any():
                    continue

                tensor_proto = torch.tensor(mean_proto, device=self.device)

                
                if label in self.global_protos:
                    self.global_protos[label] = (1 - self.prototype_momentum) * self.global_protos[label] \
                                                + self.prototype_momentum * tensor_proto
                else:
                    self.global_protos[label] = tensor_proto
            except Exception as e:
                print(f"Aggregating prototype for class {label} failed: {str(e)}")

    def _aggregate(self):
        
        
        super().aggregate_nets()

        
        self._aggregate_prototypes()