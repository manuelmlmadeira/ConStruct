import os
import pathlib
import os.path as osp

import pickle as pkl
import numpy as np
from tqdm import tqdm
from rdkit import Chem, RDLogger
import torch
from torch.utils.data import random_split
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url
from hydra.utils import get_original_cwd
import networkx as nx

from ConStruct.utils import to_dense
from ConStruct.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from ConStruct.datasets.dataset_utils import (
    load_pickle,
    save_pickle,
    Statistics,
    to_list,
    RemoveYTransform,
    compute_reference_metrics,
)
from ConStruct import metrics
from ConStruct.metrics.metrics_utils import (
    node_counts,
    atom_type_counts,
    edge_counts,
    degree_histogram,
)


class SpectreGraphDataset(InMemoryDataset):
    def __init__(
        self,
        dataset_name,
        split,
        root,
        fraction=1.0,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.sbm_file = "sbm_200.pt"
        self.planar_file = "planar_64_200.pt"
        self.comm20_file = "community_12_21_100.pt"
        self.dataset_name = dataset_name
        self.fraction = fraction

        self.split = split
        if self.split == "train":
            self.file_idx = 0
        elif self.split == "val":
            self.file_idx = 1
        else:
            self.file_idx = 2

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

        self.statistics = Statistics(
            num_nodes=load_pickle(self.processed_paths[1]),
            atom_types=torch.from_numpy(np.load(self.processed_paths[2])).float(),
            bond_types=torch.from_numpy(np.load(self.processed_paths[3])).float(),
            degree_hist=load_pickle(self.processed_paths[4]),
        )

    @property
    def raw_file_names(self):
        return ["train.pt", "val.pt", "test.pt"]

    @property
    def split_file_name(self):
        return ["train.pt", "val.pt", "test.pt"]

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        if self.split == "train":
            return [
                f"train.pt",
                f"train_n.pickle",
                f"train_atom_types.npy",
                f"train_bond_types.npy",
                f"train_degrees.pickle",
            ]
        elif self.split == "val":
            return [
                f"val.pt",
                f"val_n.pickle",
                f"val_atom_types.npy",
                f"val_bond_types.npy",
                f"val_degrees.pickle",
            ]
        else:
            return [
                f"test.pt",
                f"test_n.pickle",
                f"test_atom_types.npy",
                f"test_bond_types.npy",
                f"test_degrees.pickle",
            ]

    def download(self):
        """
        Download raw qm9 files. Taken from PyG QM9 class
        """
        if self.dataset_name == "sbm":
            raw_url = "https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/sbm_200.pt"
        elif self.dataset_name == "planar":
            raw_url = "https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/planar_64_200.pt"
        elif self.dataset_name == "comm-20":
            raw_url = "https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/community_12_21_100.pt"
        elif self.dataset_name == "ego":
            raw_url = "https://raw.githubusercontent.com/tufts-ml/graph-generation-EDGE/main/graphs/Ego.pkl"
        elif self.dataset_name in [
            "grid",
            "lobster",
            "enzymes",
            "tree",
        ]:
            raw_url = None
        else:
            raise ValueError(f"Unknown dataset {self.dataset_name}")

        if raw_url is not None:
            file_path = download_url(raw_url, self.raw_dir)

        if self.dataset_name == "ego":
            networks = pkl.load(open(file_path, "rb"))
            adjs = [
                torch.Tensor(nx.to_numpy_array(network)).fill_diagonal_(0)
                for network in networks
            ]
        elif self.dataset_name == "grid":
            graphs = []
            for i in range(10, 20):
                for j in range(10, 20):
                    graphs.append(nx.grid_2d_graph(i, j))
            adjs = [
                torch.Tensor(nx.to_numpy_array(graph)).fill_diagonal_(0)  # just in case
                for graph in graphs
            ]
        elif self.dataset_name == "enzymes":
            file_name = os.listdir(self.raw_dir)[0]
            file_path = os.path.join(self.raw_dir, file_name)
            nx_graphs = pkl.load(open(file_path, "rb"))
            adjs = [
                torch.Tensor(nx.to_numpy_array(graph)).fill_diagonal_(0)  # just in case
                for graph in nx_graphs
            ]
        elif self.dataset_name in ["tree"]:
            file_name = os.listdir(self.raw_dir)[0]
            file_path = os.path.join(self.raw_dir, file_name)
            dict_nx_graphs = pkl.load(open(file_path, "rb"))
            nx_graphs = []
            for key in ["train", "val", "test"]:  # to fix order, will be used after
                nx_graphs.extend(dict_nx_graphs[key])
            adjs = [
                torch.Tensor(nx.to_numpy_array(graph)).fill_diagonal_(0)  # just in case
                for graph in nx_graphs
            ]
        elif self.dataset_name == "lobster":
            # BIGG splits
            # file_name = os.listdir(self.raw_dir)[0]
            # file_path = os.path.join(self.raw_dir, file_name)
            # dict_nx_graphs = pkl.load(open(file_path, "rb"))
            # nx_graphs = []
            # for key in ["train", "val", "test"]:  # to fix order, will be used after
            #     nx_graphs.extend(dict_nx_graphs[key])
            # adjs = [
            #     torch.Tensor(nx.to_numpy_array(graph)).fill_diagonal_(0)  # just in case
            #     for graph in nx_graphs
            # ]
            # TODO: Delete this comment if lobster is working
            # GRAN splits
            file_name = os.listdir(self.raw_dir)[0]
            file_path = os.path.join(self.raw_dir, file_name)
            dict_nx_graphs = pkl.load(open(file_path, "rb"))
            nx_graphs = []
            for key in ["train", "test"]:
                nx_graphs.extend(dict_nx_graphs[key])
            # Conversion to torch
            adjs = [
                torch.Tensor(nx.to_numpy_array(graph)).fill_diagonal_(0)  # just in case
                for graph in nx_graphs
            ]
        # GEEL splits
        # lobster_seed = 1234 # seed from GRAN
        # npr = np.random.RandomState(lobster_seed)
        # ### load datasets
        # graphs = []
        # # synthetic graphs
        # graphs = []
        # p1 = 0.7
        # p2 = 0.7
        # count = 0
        # min_node = 10
        # max_node = 100
        # max_edge = 0
        # mean_node = 80
        # num_graphs = 100

        # seed_tmp = lobster_seed
        # while count < num_graphs:
        #     G = nx.random_lobster(mean_node, p1, p2, seed=seed_tmp)
        #     if len(G.nodes()) >= min_node and len(G.nodes()) <= max_node:
        #         graphs.append(G)
        #         if G.number_of_edges() > max_edge:
        #             max_edge = G.number_of_edges()
        #         count += 1
        #     seed_tmp += 1
        # adjs = [
        #     torch.Tensor(nx.to_numpy_array(graph)).fill_diagonal_(0)  # just in case
        #     for graph in graphs
        # ]
        else:
            (
                adjs,
                eigvals,
                eigvecs,
                n_nodes,
                max_eigval,
                min_eigval,
                same_sample,
                n_max,
            ) = torch.load(file_path)

        # Planarity checks
        # are_planar = [nx.is_planar(nx.from_numpy_array(adj.numpy())) for adj in adjs]
        # print(are_planar)
        # print("Is the dataset composed of just planar graphs? ", all(are_planar))
        # is_planar = True
        # for i, adj in enumerate(adjs):
        #     if not nx.is_planar(nx.from_numpy_array(adj.numpy())):
        #         is_planar = False
        #         print(f"Graph {i} is not planar")
        #         break
        # breakpoint()

        self.num_graphs = len(adjs)

        if self.dataset_name == "ego":
            test_len = int(round(self.num_graphs * 0.2))
            train_len = int(round(self.num_graphs * 0.8))
            val_len = int(round(self.num_graphs * 0.2))
            g_cpu = torch.Generator().manual_seed(0)
            indices = torch.randperm(self.num_graphs, generator=g_cpu)
            train_indices = indices[:train_len]
            val_indices = indices[:val_len]
            test_indices = indices[train_len:]
        elif self.dataset_name in ["tree"]:
            # Lobster: BIGG splits
            train_len = len(dict_nx_graphs["train"])
            val_len = len(dict_nx_graphs["val"])
            test_len = len(dict_nx_graphs["test"])
            indices = torch.range(0, self.num_graphs - 1)
            train_indices = indices[:train_len]
            val_indices = indices[train_len : train_len + val_len]
            test_indices = indices[train_len + val_len :]
        elif self.dataset_name == "lobster":
            # GRAN splits (check if similar to Spectre split - it seems so except for test that is predefined)
            test_len = 20
            val_len = 16
            train_len = 64
            assert train_len + val_len + test_len == self.num_graphs
            indices = list(range(self.num_graphs - test_len))
            # npr = np.random.RandomState(1234) # seed from GRAN
            # npr.shuffle(indices)  # does not work with tensor
            indices = torch.Tensor(indices)
            # val_indices = indices[:val_len]
            # train_indices = indices[val_len : val_len + train_len]
            val_indices = indices[train_len : train_len + val_len]
            train_indices = indices[:train_len]
            test_indices = torch.range(self.num_graphs - test_len, self.num_graphs)

        # TODO: delete comments below if lobster is working
        # first splits (val ref metrics for orbit >> test ref metrics for orbit, causes problems)
        #     test_len = int(round(self.num_graphs * 0.2))
        #     train_len = int(round((self.num_graphs - test_len) * 0.8))
        #     val_len = self.num_graphs - train_len - test_len
        #     indices = torch.range(0, self.num_graphs - 1)
        #     val_indices = indices[:val_len]
        #     train_indices = indices[val_len : val_len + train_len]
        #     # train_indices = indices[:train_len]
        #     # val_indices = indices[train_len : train_len + val_len]
        #     test_indices = indices[val_len + train_len :]
        # GEEL splits
        # train = data[int(val_size*len(data)):int((train_size+val_size)*len(data))]
        # val = data[:int(val_size*len(data))]
        # test = data[int((train_size+val_size)*len(data)):]
        # test_frac = 0.2
        # val_frac = 0.16
        # train_frac = 1 - test_frac - val_frac
        # test_len = int(round(self.num_graphs * test_frac))
        # train_len = int(round(self.num_graphs * train_frac))
        # val_len = self.num_graphs - train_len - test_len
        # indices = list(range(self.num_graphs))
        # # npr = np.random.RandomState(lobster_seed)
        # # npr.shuffle(indices)  # does not work with tensor
        # indices = torch.Tensor(indices)
        # val_indices = indices[:val_len]
        # train_indices = indices[val_len : val_len + train_len]
        # test_indices = indices[val_len + train_len :]
        else:
            test_len = int(round(self.num_graphs * 0.2))
            train_len = int(round((self.num_graphs - test_len) * 0.8))
            val_len = self.num_graphs - train_len - test_len
            indices = torch.randperm(
                self.num_graphs, generator=torch.Generator().manual_seed(1234)
            )  # Use this generator to follow Spectre paper splits
            train_indices = indices[:train_len]
            val_indices = indices[train_len : train_len + val_len]
            test_indices = indices[train_len + val_len :]

        train_data = []
        val_data = []
        test_data = []

        for i, adj in enumerate(adjs):
            if i in train_indices:
                train_data.append(adj)
            if i in val_indices:
                val_data.append(adj)
            if i in test_indices:
                test_data.append(adj)
            if (
                i not in train_indices
                and i not in val_indices
                and i not in test_indices
            ):
                raise ValueError(f"Index {i} not in any split")

        # Reduce trainind dataset size according to fraction
        train_len = round(self.fraction * train_len)
        train_data = train_data[:train_len]
        print(f"Dataset sizes: train {train_len}, val {val_len}, test {test_len}")

        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])

    def process(self):
        raw_dataset = torch.load(os.path.join(self.raw_dir, "{}.pt".format(self.split)))
        data_list = []
        for adj in raw_dataset:
            n = adj.shape[-1]
            X = torch.zeros(n, dtype=torch.long)
            edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
            edge_attr = torch.ones(edge_index.shape[-1], dtype=torch.long)
            n_nodes = n * torch.ones(1, dtype=torch.long)
            data = torch_geometric.data.Data(
                x=X, edge_index=edge_index, edge_attr=edge_attr, n_nodes=n_nodes
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        num_nodes = node_counts(data_list)
        atom_types = atom_type_counts(data_list, num_classes=1)
        bond_types = edge_counts(data_list, num_bond_types=2)
        degrees_hist = degree_histogram(data_list)
        torch.save(self.collate(data_list), self.processed_paths[0])
        save_pickle(num_nodes, self.processed_paths[1])
        np.save(self.processed_paths[2], atom_types)
        np.save(self.processed_paths[3], bond_types)
        save_pickle(degrees_hist, self.processed_paths[4])


class SpectreGraphDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_name = self.cfg.dataset.name
        self.datadir = (
            cfg.dataset.datadir
            if cfg.dataset.fraction == 1.0
            else cfg.dataset.datadir + f"_{cfg.dataset.fraction}"
        )
        base_path = pathlib.Path(get_original_cwd()).parents[0]
        root_path = os.path.join(base_path, self.datadir)
        transform = RemoveYTransform()

        datasets = {
            "train": SpectreGraphDataset(
                dataset_name=self.cfg.dataset.name,
                transform=transform,
                split="train",
                root=root_path,
                fraction=self.cfg.dataset.fraction,
            ),
            "val": SpectreGraphDataset(
                dataset_name=self.cfg.dataset.name,
                transform=transform,
                split="val",
                root=root_path,
            ),
            "test": SpectreGraphDataset(
                dataset_name=self.cfg.dataset.name,
                transform=transform,
                split="test",
                root=root_path,
            ),
        }

        self.statistics = {
            "train": datasets["train"].statistics,
            "val": datasets["val"].statistics,
            "test": datasets["test"].statistics,
        }

        super().__init__(cfg, datasets)
        super().prepare_dataloader()
        self.inner = self.train_dataset


class SpectreDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule):
        self.is_molecular = False
        self.is_tls = False
        self.dataset_name = datamodule.dataset_name
        self.atom_types = datamodule.inner.statistics.atom_types
        self.bond_types = datamodule.inner.statistics.bond_types
        self.statistics = datamodule.statistics

        super().complete_infos(datamodule.statistics)
        compute_reference_metrics(self, datamodule)
