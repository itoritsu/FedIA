from models.utils.federated_model import FederatedModel
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

class FedSage(FederatedModel):
    NAME = 'fedsage'

    def __init__(self, nets_list, args, transform):
        super(FedSage, self).__init__(nets_list, args, transform)
        self.global_net = copy.deepcopy(nets_list[0])

    def ini(self):
        global_w = self.global_net.state_dict()
        for net in self.nets_list:
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i])

        self.aggregate_nets(None)

    def _train_net(self, index, net, graph_data):
        """
        graph_data : torch_geometric.data.Data
                     Here priv_dataset returns a single graph rather than a DataLoader.
        """
        net = net.to(self.device).train()
        graph_data = graph_data.to(self.device)  

        optimizer = optim.SGD(net.parameters(), lr=self.local_lr,
                              momentum=0.9, weight_decay=1e-5)
        criterion = F.nll_loss

        iter_bar = tqdm(range(self.local_epoch), desc=f'Client {index}')
        grads = {n: torch.zeros_like(p, device=self.device)
                 for n, p in net.named_parameters()}

        for _ in iter_bar:
            optimizer.zero_grad()

            
            out = net(graph_data.x, graph_data.edge_index)
            loss = criterion(out[graph_data.train_mask],
                             graph_data.y[graph_data.train_mask])
            loss.backward()

            
            for name, p in net.named_parameters():
                if p.grad is not None:
                    grads[name] += p.grad

            optimizer.step()
            iter_bar.set_description(f'Client {index} loss={loss.item():.4f}')

        
        for n in grads:
            grads[n] /= self.local_epoch
        self.local_gradients[index] = grads
