


import os, urllib.request, numpy as np, torch
from pathlib import Path
from torch_geometric.data import InMemoryDataset, Data
from datasets.utils.federated_dataset import FederatedDataset
from datasets.utils.splitter import RandomSplitter_wikinet         

NPZ_URL_TMPL = "https://graphmining.ai/datasets/ptg/wiki/{}.npz"
DOMAINS = ["chameleon", "squirrel", "crocodile"]


class WikiNetRawNPZ(InMemoryDataset):
    def __init__(self, root:str, domain:str, max_dim=None,
                 transform=None, pre_transform=None):
        self.domain, self._max_dim = domain, max_dim
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self):
        
        return Path(self.root) / "WikiNet_npz_process" / self.domain

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download_npz(self, save_path, domain):
        url = NPZ_URL_TMPL.format(domain)
        print(f"[WikiNet] downloading {url} → {save_path}")
        urllib.request.urlretrieve(url, save_path)

    def process(self):
        raw_dir = Path(self.root) / "WikiNet_npz_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        npz_path = raw_dir / f"{self.domain}.npz"
        if not npz_path.exists(): self.download_npz(npz_path, self.domain)

        npz = np.load(npz_path)
        x = torch.from_numpy(npz["features"]).float()
        y = torch.from_numpy(npz["target"]).long()
        ei = torch.from_numpy(npz["edges"]).long().t().contiguous()

        
        if self._max_dim is not None and x.size(1) < self._max_dim:
            pad = torch.zeros((x.size(0), self._max_dim - x.size(1)))
            x = torch.cat([x, pad], 1)
        mask = torch.ones(x.size(0), dtype=torch.bool)
        data = Data(x=x, edge_index=ei, y=y,
                    train_mask=mask, val_mask=mask, test_mask=mask)
        torch.save(self.collate([data]), self.processed_paths[0])


class FedWikiNetMusae(FederatedDataset):
    NAME, SETTING, DOMAINS_LIST = "fl_wikinet", "domain_skew", DOMAINS
    domain_dict = {"chameleon": 2, "squirrel": 2, 'crocodile': 2}

    def __init__(self, args):
        super().__init__(args)

        
        raw_root = Path(args.data_root) / "WikiNet_npz_raw"
        global_dim = 0
        for d in self.DOMAINS_LIST:
            npz_path = raw_root / f"{d}.npz"
            if not npz_path.exists():
                urllib.request.urlretrieve(NPZ_URL_TMPL.format(d), npz_path)
            with np.load(npz_path, mmap_mode="r") as npz:
                global_dim = max(global_dim, npz["features"].shape[1])

        
        self.datasets = {
            d: WikiNetRawNPZ(root=args.data_root, domain=d, max_dim=global_dim)
            for d in self.DOMAINS_LIST
        }
        self.N_CLASS = max(int(ds[0].y.max()) for ds in self.datasets.values()) + 1

    @staticmethod
    def get_transform(): return []

    
    def get_data_loaders(self, domain_distribution=None):
        train, test = [], []
        for d in self.DOMAINS_LIST:
            cnt = int(domain_distribution.get(d, self.domain_dict[d])) if domain_distribution else self.domain_dict[d]
            g_full = self.datasets[d][0]
            splitter = RandomSplitter_wikinet(client_num=cnt)
            clients, global_g = splitter(g_full)
            for i, cg in enumerate(clients):
                cg.client_id, cg.domain, cg.data_name = f"{d}_c{i}", d, d
                train.append(cg)
            global_g.client_id, global_g.domain, global_g.data_name = f"{d}_global", d, d
            test.append(global_g)
        print("Total clients:", len(train))
        return train, test

    def get_backbone(self, parti_num, names_list):
        nets = self.get_gnn_backbone_dict()
        in_dim = self.datasets[self.DOMAINS_LIST[0]][0].x.shape[1]
        return [nets[n](in_dim, self.N_CLASS, self.args.hidden) for n in names_list]
