import torch

from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.utils import add_self_loops, remove_self_loops, \
    to_undirected
from torch_geometric.utils import dropout_adj, index_to_mask
import numpy as np
import networkx as nx
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
import torch
import math

import community as community_louvain

EPSILON = 1e-5

def split_train_test(data, train_rate, test_rate):
    random_node_indices = np.random.permutation(data.y.shape[0])
    training_size = int(len(random_node_indices) * 0.1)
    val_size = int(len(random_node_indices) * 0.1)
    train_node_indices = random_node_indices[:training_size]
    val_node_indices = random_node_indices[training_size:training_size + val_size]
    test_node_indices = random_node_indices[training_size + val_size:]

    train_masks = torch.zeros([data.y.shape[0]], dtype=torch.uint8)
    train_masks[train_node_indices] = 1
    train_masks = train_masks.bool()
    val_masks = torch.zeros([data.y.shape[0]], dtype=torch.uint8)
    val_masks[val_node_indices] = 1
    val_masks = val_masks.bool()
    test_masks = torch.zeros([data.y.shape[0]], dtype=torch.uint8)
    test_masks[test_node_indices] = 1
    test_masks = test_masks.bool()

    data.train_mask = train_masks
    data.val_mask = val_masks
    data.test_mask = test_masks
    return data
class RandomSplitter(BaseTransform):
    
    def __init__(self,
                 client_num,
                 sampling_rate=None,
                 overlapping_rate=0,
                 drop_edge=0,missing_link = 0,percent=30):
        self.client_num = client_num
        self.client_num_need = client_num
        self.ovlap = overlapping_rate
        if sampling_rate is not None:
            self.sampling_rate = np.array(
                [float(val) for val in sampling_rate.split(',')])
        else:
            
            self.sampling_rate = (np.ones(self.client_num) -
                                  self.ovlap) / self.client_num

        if len(self.sampling_rate) != self.client_num:
            raise ValueError(
                f'The client_num ({self.client_num}) should be equal to the '
                f'lenghth of sampling_rate and overlapping_rate.')

        if abs((sum(self.sampling_rate) + self.ovlap) - 1) > EPSILON:
            raise ValueError(
                f'The sum of sampling_rate:{self.sampling_rate} and '
                f'overlapping_rate({self.ovlap}) should be 1.')

        self.drop_edge = drop_edge
        self.missing_link = missing_link

    def __call__(self, data, global_dataset, **kwargs):
        if self.missing_link > 0:
            print(data.num_edges)
            data.edge_index, _ = dropout_adj(data.edge_index, p=self.missing_link, force_undirected=True,
                                                   num_nodes=data.num_nodes)
            print(data.num_edges)
        data.index_orig = torch.arange(data.num_nodes)
        G = to_networkx(
            data,
            node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'],
            to_undirected=True)
        nx.set_node_attributes(G,
                               dict([(nid, nid)
                                     for nid in range(nx.number_of_nodes(G))]),
                               name="index_orig")

        client_node_idx = {idx: [] for idx in range(self.client_num)}

        indices = data.index_orig.cpu().numpy()
        
        sum_rate = 0
        for idx, rate in enumerate(self.sampling_rate):
            client_node_idx[idx] = indices[round(sum_rate *
                                                 data.num_nodes):round(
                                                     (sum_rate + rate) *
                                                     data.num_nodes)]
            sum_rate += rate

        if self.ovlap:
            ovlap_nodes = indices[round(sum_rate * data.num_nodes):]
            for idx in client_node_idx:
                client_node_idx[idx] = np.concatenate(
                    (client_node_idx[idx], ovlap_nodes))

        
        if self.drop_edge:
            ovlap_graph = nx.Graph(nx.subgraph(G, ovlap_nodes))
            ovlap_edge_ind = np.random.permutation(
                ovlap_graph.number_of_edges())
            drop_all = ovlap_edge_ind[:round(ovlap_graph.number_of_edges() *
                                             self.drop_edge)]
            drop_client = [
                drop_all[s:s + round(len(drop_all) / self.client_num)]
                for s in range(0, len(drop_all),
                               round(len(drop_all) / self.client_num))
            ]

        graphs = []
        for owner in client_node_idx:
            nodes = client_node_idx[owner]
            sub_g = nx.DiGraph(nx.subgraph(G, nodes))
            if self.drop_edge:
                sub_g.remove_edges_from(
                    np.array(ovlap_graph.edges)[drop_client[owner]])
            graphs.append(from_networkx(sub_g))

        dataset = [ds for ds in graphs]
        client_num = min(len(dataset), self.client_num
                         ) if self.client_num > 0 else len(dataset)


        graphs = []
        for client_idx in range(1, len(dataset) + 1):
            local_data = dataset[client_idx - 1]
            
            local_data.edge_index = add_self_loops(
                to_undirected(remove_self_loops(local_data.edge_index)[0]),
                num_nodes=local_data.x.shape[0])[0]
            graphs.append(local_data)
        if global_dataset is not None:
            global_graph = global_dataset
            train_mask = torch.zeros_like(global_graph.train_mask)
            val_mask = torch.zeros_like(global_graph.val_mask)
            test_mask = torch.zeros_like(global_graph.test_mask)

            for client_subgraph in graphs:
                train_mask[client_subgraph.index_orig[
                    client_subgraph.train_mask]] = True
                val_mask[client_subgraph.index_orig[
                    client_subgraph.val_mask]] = True
                test_mask[client_subgraph.index_orig[
                    client_subgraph.test_mask]] = True
            global_graph.train_mask = train_mask
            global_graph.val_mask = val_mask
            global_graph.test_mask = test_mask
        global_dataset = global_graph
        graphs = graphs[:self.client_num_need]
        return graphs,global_dataset



class LouvainSplitter():
    
    def __init__(self, client_num, delta=20):
        self.delta = delta
        self.client_num = client_num

    def __call__(self, data, **kwargs):
        data.index_orig = torch.arange(data.num_nodes)
        G = to_networkx(
            data,
            node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'],
            to_undirected=True)
        nx.set_node_attributes(G,
                               dict([(nid, nid)
                                     for nid in range(nx.number_of_nodes(G))]),
                               name="index_orig")
        partition = community_louvain.best_partition(G)

        cluster2node = {}
        for node in partition:
            cluster = partition[node]
            if cluster not in cluster2node:
                cluster2node[cluster] = [node]
            else:
                cluster2node[cluster].append(node)

        max_len = len(G) // self.client_num - self.delta
        max_len_client = len(G) // self.client_num

        tmp_cluster2node = {}
        for cluster in cluster2node:
            while len(cluster2node[cluster]) > max_len:
                tmp_cluster = cluster2node[cluster][:max_len]
                tmp_cluster2node[len(cluster2node) + len(tmp_cluster2node) +
                                 1] = tmp_cluster
                cluster2node[cluster] = cluster2node[cluster][max_len:]
        cluster2node.update(tmp_cluster2node)

        orderedc2n = (zip(cluster2node.keys(), cluster2node.values()))
        orderedc2n = sorted(orderedc2n, key=lambda x: len(x[1]), reverse=True)

        client_node_idx = {idx: [] for idx in range(self.client_num)}
        idx = 0
        for (cluster, node_list) in orderedc2n:
            while len(node_list) + len(
                    client_node_idx[idx]) > max_len_client + self.delta:
                idx = (idx + 1) % self.client_num
            client_node_idx[idx] += node_list
            idx = (idx + 1) % self.client_num

        graphs = []
        for owner in client_node_idx:
            nodes = client_node_idx[owner]
            graphs.append(from_networkx(nx.subgraph(G, nodes)))

        graphs = graphs[:self.client_num_need]
        return graphs


class RandomSplitter4JODIE:
    def __init__(self, client_num, min_nodes=20):
        self.client_num = min(client_num, 10)
        self.min_nodes = max(min_nodes, 10)
        self.edge_overlap_rate = 0.1  

    def _remap_node_indices(self, sub_edges):
        
        
        unique_nodes, inverse_indices = torch.unique(sub_edges, return_inverse=True)

        
        node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_nodes.tolist())}

        
        remapped_edges = inverse_indices.reshape(2, -1)

        
        max_idx = remapped_edges.max().item()
        assert max_idx < len(unique_nodes), f"Index out of range: {max_idx} >= {len(unique_nodes)}"

        return remapped_edges, unique_nodes

    def __call__(self, data):
        edge_index = data.edge_index
        total_edges = edge_index.size(1)

        
        overlap_edges = int(total_edges * self.edge_overlap_rate)
        main_edges = total_edges - overlap_edges

        
        edges_per_client = max(main_edges // self.client_num, 1000)
        client_graphs = []

        for i in range(self.client_num):
            
            start = i * edges_per_client
            end = min((i + 1) * edges_per_client, main_edges)

            if start >= main_edges:
                break

            
            overlap_start = main_edges + i * (overlap_edges // self.client_num)
            overlap_end = main_edges + (i + 1) * (overlap_edges // self.client_num)

            sub_edges = torch.cat([
                edge_index[:, start:end],
                edge_index[:, overlap_start:overlap_end]
            ], dim=1)

            
            remapped_edges, nodes = self._remap_node_indices(sub_edges)

            
            if len(nodes) < self.min_nodes:
                padding = nodes.repeat(self.min_nodes // len(nodes) + 1)[:self.min_nodes]
                nodes = torch.cat([nodes, padding])
                print(f"Client {i} lacks nodes; padding to {self.min_nodes}")

            
            client_data = Data(
                x=data.x[nodes],
                edge_index=remapped_edges,
                y=data.y[nodes],
                train_mask=data.train_mask[nodes],
                val_mask=data.val_mask[nodes],
                test_mask=data.test_mask[nodes]
            )
            client_graphs.append(client_data)

        return client_graphs[:self.client_num], data

from sklearn.cluster import KMeans

class KMeansSplitter:
    
    def __init__(self, client_num, random_state=42):
        self.client_num = client_num
        self.random_state = random_state

    def __call__(self, data_obj):
        num_nodes = data_obj.num_nodes
        
        feats = data_obj.x.cpu().numpy()
        km = KMeans(n_clusters=self.client_num, random_state=self.random_state).fit(feats)
        labels = km.labels_
        client_graphs = []

        
        for i in range(self.client_num):
            nodes = torch.where(torch.tensor(labels) == i)[0]
            
            edge_index = data_obj.edge_index
            mask_u = torch.isin(edge_index[0], nodes)
            mask_v = torch.isin(edge_index[1], nodes)
            edge_mask = mask_u & mask_v
            sub_ei = edge_index[:, edge_mask]

            
            id_map = {int(n): idx for idx, n in enumerate(nodes.tolist())}
            remapped = torch.tensor([
                [id_map[int(u)], id_map[int(v)]]
                for u, v in sub_ei.t()
            ]).t().contiguous()

            client_data = Data(
                x=data_obj.x[nodes],
                edge_index=remapped,
                y=data_obj.y[nodes],
                train_mask=data_obj.train_mask[nodes],
                val_mask=data_obj.val_mask[nodes],
                test_mask=data_obj.test_mask[nodes],
            )
            client_graphs.append(client_data)

        return client_graphs

class RandomSplitter_wikinet:
    r
    def __init__(self,
                 client_num: int,
                 train_ratio: float = 0.6,
                 val_ratio: float = 0.2,
                 shuffle: bool = True,
                 seed: int | None = None,
                 edge_transform: bool = True):

        assert 0 < train_ratio < 1 and 0 < val_ratio < 1 \
               and train_ratio + val_ratio < 1, \
               "train+val must < 1"

        self.client_num   = int(client_num)
        self.train_ratio  = train_ratio
        self.val_ratio    = val_ratio
        self.shuffle      = shuffle
        self.seed         = seed
        self.edge_transform = edge_transform

    
    def _randperm(self, N: int) -> torch.Tensor:
        
        idx = torch.arange(N)
        if self.shuffle:
            gen = torch.Generator()
            if self.seed is not None:
                gen.manual_seed(self.seed)
            idx = idx[torch.randperm(N, generator=gen)]
        return idx

    
    def _split_masks(self, n_nodes: int, gen: torch.Generator) -> tuple[torch.Tensor, ...]:
        
        n_train = math.floor(n_nodes * self.train_ratio)
        n_val   = math.floor(n_nodes * self.val_ratio)
        idx_perm = torch.randperm(n_nodes, generator=gen)

        train_mask          = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[idx_perm[:n_train]] = True
        val_mask            = torch.zeros_like(train_mask)
        val_mask[idx_perm[n_train:n_train + n_val]] = True
        test_mask = ~(train_mask | val_mask)
        return train_mask, val_mask, test_mask

    
    def __call__(self, data: Data):
        
        global_graph = data

        
        N       = data.num_nodes
        idx_all = self._randperm(N)
        
        sizes = [N // self.client_num + (1 if i < N % self.client_num else 0)
                 for i in range(self.client_num)]
        node_splits = torch.split(idx_all, sizes)

        
        client_graphs = []
        gen_mask = torch.Generator()
        if self.seed is not None:
            gen_mask.manual_seed(self.seed + 10086)  

        
        g_train = torch.zeros(N, dtype=torch.bool)
        g_val   = torch.zeros_like(g_train)
        g_test  = torch.zeros_like(g_train)

        for nodes in node_splits:
            
            ei_sub, _ = subgraph(nodes, data.edge_index,
                                 relabel_nodes=True, num_nodes=N)

            if self.edge_transform:
                ei_sub = to_undirected(ei_sub, num_nodes=len(nodes))
                ei_sub, _ = add_self_loops(ei_sub, num_nodes=len(nodes))

            x_sub, y_sub = data.x[nodes], data.y[nodes]
            tm, vm, sm   = self._split_masks(len(nodes), gen_mask)

            sub_data = Data(x=x_sub,
                            edge_index=ei_sub,
                            y=y_sub,
                            train_mask=tm,
                            val_mask=vm,
                            test_mask=sm)
            client_graphs.append(sub_data)

            
            g_train[nodes] = tm
            g_val[nodes]   = vm
            g_test[nodes]  = sm

        
        global_graph.train_mask = g_train
        global_graph.val_mask   = g_val
        global_graph.test_mask  = g_test

        return client_graphs, global_graph