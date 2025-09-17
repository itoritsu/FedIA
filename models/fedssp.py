import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy
from models.utils.federated_model import FederatedModel
from utils.util import diff_loss

class FedSSP(FederatedModel):
    NAME = 'fedssp'
    COMPATIBILITY = ['heterogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedSSP, self).__init__(nets_list, args, transform)
        self.global_spectral_weights = None
        self.local_preferences = {}

    def ini(self):
        """Initialize global model and distribute to local clients."""
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.global_net.state_dict()
        for net in self.nets_list:
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        """Perform local updates for selected clients."""
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

        for client_idx in online_clients:
            self._train_client(client_idx, self.nets_list[client_idx], priloader_list[client_idx])

        self.aggregate_nets()

    def _train_client(self, client_idx, net, train_loader):
        """Train a single client locally."""
        net = net.to(self.device)
        train_loader = train_loader.to(self.device)
        optimizer = optim.AdamW(net.parameters(), lr=self.local_lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        # Initialize local preference adjustment
        preference_module = nn.Parameter(torch.zeros_like(next(net.parameters())))
        optimizer_pref = optim.SGD([preference_module], lr=self.local_lr)

        net.train()
        for epoch in range(self.local_epoch):
            for data in train_loader:
                optimizer.zero_grad()
                optimizer_pref.zero_grad()

                # Forward pass
                out = net(data)
                loss = criterion(out, data.y)

                # Adjust with preference
                adjusted_out = out + preference_module
                adjusted_loss = criterion(adjusted_out, data.y)

                # Regularize preference adjustments
                reg_loss = torch.norm(preference_module)
                total_loss = adjusted_loss + self.args.pref_lambda * reg_loss

                # Backpropagation
                total_loss.backward()
                optimizer.step()
                optimizer_pref.step()

        # Save local preferences
        self.local_preferences[client_idx] = preference_module.clone()

    def aggregate_nets(self):
        """Aggregate models from selected clients."""
        global_weights = {k: torch.zeros_like(v) for k, v in self.global_net.state_dict().items()}

        # Direct averaging of local updates
        for client_idx in self.online_clients:
            local_weights = self.nets_list[client_idx].state_dict()
            for k in global_weights:
                global_weights[k] += local_weights[k] / len(self.online_clients)

        self.global_net.load_state_dict(global_weights)

        # Share spectral knowledge globally
        self.global_spectral_weights = {name: param.clone() for name, param in self.global_net.named_parameters() if 'spectral' in name}

        # Update local models with global weights
        for net in self.nets_list:
            net.load_state_dict(self.global_net.state_dict())
            if self.global_spectral_weights:
                for name, param in net.named_parameters():
                    if name in self.global_spectral_weights:
                        param.data.copy_(self.global_spectral_weights[name])
