import copy
import os
import os.path as osp
import pathlib


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
import matplotlib.pyplot as plt
from matplotlib import rc

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


class TLSGraphDataset(InMemoryDataset):
    def __init__(
        self,
        dataset_name,
        num_graphs,
        split,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.dataset_name = dataset_name
        self.num_graphs = num_graphs

        self.split = split
        # self.file_idx is used by the init of super class
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
        return ["train.pkl", "val.pkl", "test.pkl"]

    @property
    def split_file_name(self):
        return ["train.pkl", "val.pkl", "test.pkl"]

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip splitting."""
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
        full_dataset_path = ""
        full_dataset = load_pickle(full_dataset_path)
        if self.dataset_name == "low_tls":
            dataset = full_dataset["low_TLS_graphs"]
        elif self.dataset_name == "high_tls":
            dataset = full_dataset["high_TLS_graphs"]
        else:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}")
        # Select only graphs with number of nodes in a range
        nx_graphs = []
        min_num_nodes = 20
        max_num_nodes = 81  # fits GPU

        for graph in dataset:
            if len(nx_graphs) >= self.num_graphs:
                break
            has_right_nodes = min_num_nodes <= graph.number_of_nodes() <= max_num_nodes
            is_not_isomorphic = True
            cg_new = CellGraph(graph)
            for previous_graph in nx_graphs:
                if cg_new.is_isomorphic(CellGraph(previous_graph)):
                    is_not_isomorphic = False
                    print("Found isomorphic graphs")
                    break
            if has_right_nodes and is_not_isomorphic:
                nx_graphs.append(graph)

        assert self.num_graphs == len(nx_graphs)

        # Planarity checks
        # are_planar = [nx.is_planar(nx_graph) for nx_graph in nx_graphs]
        # print(are_planar)
        # print("Is the dataset composed of just planar graphs? ", all(are_planar))
        # breakpoint()
        # is_planar = True
        # for i, adj in enumerate(adjs):
        #     if not nx.is_planar(nx.from_numpy_array(adj.numpy())):
        #         is_planar = False
        #         print(f"Graph {i} is not planar")
        #         break
        # breakpoint()

        test_len = int(round(self.num_graphs * 0.2))
        train_len = int(round((self.num_graphs - test_len) * 0.8))
        val_len = self.num_graphs - train_len - test_len
        indices = torch.randperm(
            self.num_graphs, generator=torch.Generator().manual_seed(1234)
        )
        train_indices = indices[:train_len]
        val_indices = indices[train_len : train_len + val_len]
        test_indices = indices[train_len + val_len :]

        train_data = []
        val_data = []
        test_data = []

        for i, nx_graph in enumerate(nx_graphs):
            if i in train_indices:
                train_data.append(nx_graph)
            if i in val_indices:
                val_data.append(nx_graph)
            if i in test_indices:
                test_data.append(nx_graph)
            if (
                i not in train_indices
                and i not in val_indices
                and i not in test_indices
            ):
                raise ValueError(f"Index {i} not in any split")

        print(f"Dataset sizes: train {train_len}, val {val_len}, test {test_len}")
        save_pickle(train_data, self.raw_paths[0])
        save_pickle(val_data, self.raw_paths[1])
        save_pickle(test_data, self.raw_paths[2])

    def process(self):
        raw_dataset = load_pickle(
            os.path.join(self.raw_dir, "{}.pkl".format(self.split))
        )

        data_list = []
        for graph in raw_dataset:
            cell_graph = CellGraph(graph)
            data = cell_graph.to_torch_geometric()

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        num_nodes = node_counts(data_list)
        atom_types = atom_type_counts(data_list, num_classes=len(PHENOTYPE_DECODER))
        bond_types = edge_counts(data_list, num_bond_types=2)
        degrees_hist = degree_histogram(data_list)
        torch.save(self.collate(data_list), self.processed_paths[0])
        save_pickle(num_nodes, self.processed_paths[1])
        np.save(self.processed_paths[2], atom_types)
        np.save(self.processed_paths[3], bond_types)
        save_pickle(degrees_hist, self.processed_paths[4])


class TLSDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_name = self.cfg.dataset.name
        self.datadir = cfg.dataset.datadir + f"_{cfg.dataset.num_graphs}"
        base_path = pathlib.Path(get_original_cwd()).parents[0]
        root_path = os.path.join(base_path, self.datadir)
        transform = RemoveYTransform()

        datasets = {
            "train": TLSGraphDataset(
                dataset_name=self.cfg.dataset.name,
                transform=transform,
                split="train",
                root=root_path,
                num_graphs=self.cfg.dataset.num_graphs,
            ),
            "val": TLSGraphDataset(
                dataset_name=self.cfg.dataset.name,
                transform=transform,
                split="val",
                root=root_path,
                num_graphs=self.cfg.dataset.num_graphs,
            ),
            "test": TLSGraphDataset(
                dataset_name=self.cfg.dataset.name,
                transform=transform,
                split="test",
                root=root_path,
                num_graphs=self.cfg.dataset.num_graphs,
            ),
        }

        # Isomorphism check
        # print("Checking isomorphism in generated")
        # seen_cgs = []
        # for key in datasets.keys():
        #     print(f"Checking {key} dataset (len: {len(datasets[key])})")
        #     for tg_graph in datasets[key]:
        #         cg_new = CellGraph.from_torch_geometric(tg_graph)
        #         for seen_cg in seen_cgs:
        #             if cg_new.is_isomorphic(seen_cg):
        #                 raise ValueError("Found isomorphic graphs")
        #         seen_cgs.append(cg_new)
        # print(f"Dataset contains {len(seen_cgs)} non-isomorphic graphs")

        # Number of nodes check
        # num_nodes_list = []
        # for key in datasets.keys():
        #     for tg_graph in datasets[key]:
        #         num_nodes_list.append(tg_graph.num_nodes)
        # print(f"Minumum number of nodes: {min(num_nodes_list)}")
        # print(f"Maximum number of nodes: {max(num_nodes_list)}")
        # print(f"Mean number of nodes: {np.mean(num_nodes_list)}")
        # breakpoint()

        # Check connectedness
        # for key in datasets.keys():
        #     print(
        #         f"Checking graph connectedness in {key} dataset (len: {len(datasets[key])})"
        #     )
        #     for tg_graph in datasets[key]:
        #         cg_new = CellGraph.from_torch_geometric(tg_graph)
        #         if not nx.is_connected(cg_new):
        #             raise ValueError("Found disconnected graphs")
        #     print(f"{key} dataset contains only connected graphs")
        # breakpoint()

        self.statistics = {
            "train": datasets["train"].statistics,
            "val": datasets["val"].statistics,
            "test": datasets["test"].statistics,
        }

        super().__init__(cfg, datasets)
        super().prepare_dataloader()
        self.inner = self.train_dataset


class TLSInfos(AbstractDatasetInfos):
    def __init__(self, datamodule):
        self.is_molecular = False
        self.is_tls = True
        self.dataset_name = datamodule.dataset_name
        self.atom_types = datamodule.inner.statistics.atom_types
        self.bond_types = datamodule.inner.statistics.bond_types
        self.statistics = datamodule.statistics

        super().complete_infos(datamodule.statistics)
        compute_reference_metrics(self, datamodule)


PHENOTYPE_DECODER = [
    "B",
    "T",
    "Epithelial",
    "Fibroblast",
    "Myofibroblast",
    "CD38+ Lymphocyte",
    "Macrophages/Granulocytes",
    "Marker",
    "Endothelial",
]

PHENOTYPE_ENCODER = {v: k for k, v in enumerate(PHENOTYPE_DECODER)}


class CellGraph(nx.Graph):
    def __init__(self, graph):
        # nx specific
        super().__init__()
        self.add_nodes_from(graph.nodes(data=True))
        self.add_edges_from(graph.edges(data=True))

        self.tls_features = self.compute_tls_features()

    def has_low_TLS(self):
        return self.tls_features["k_1"] < 0.05

    def has_high_TLS(self):
        return 0.05 < self.tls_features["k_2"]

    def to_torch_geometric(self):
        n = self.number_of_nodes()
        n_nodes = n * torch.ones(1, dtype=torch.long)
        phenotypes = [self.nodes[node].get("phenotype") for node in self.nodes()]
        encoded_phenotypes = [PHENOTYPE_ENCODER[phenotype] for phenotype in phenotypes]
        X = torch.tensor(encoded_phenotypes, dtype=torch.long)
        torch_adj = torch.Tensor(
            nx.to_numpy_array(self)
        )  # by default follows same order as G.nodes()
        edge_index, _ = torch_geometric.utils.dense_to_sparse(torch_adj)
        edge_attr = torch.ones(edge_index.shape[-1], dtype=torch.long)  # no edge types

        data = torch_geometric.data.Data(
            x=X,
            edge_index=edge_index,
            edge_attr=edge_attr,
            n_nodes=n_nodes,
        )
        return data

    @classmethod
    def from_torch_geometric(cls, data):
        nx_graph = torch_geometric.utils.to_networkx(
            data, node_attrs=["x"], to_undirected=True
        )

        for node in nx_graph.nodes():
            encoded_phenotype = nx_graph.nodes[node].pop(
                "x"
            )  # also deletes the key "x"
            nx_graph.nodes[node]["phenotype"] = PHENOTYPE_DECODER[encoded_phenotype]

        cell_graph = cls(nx_graph)
        return cell_graph

    @classmethod
    def from_placeholder(cls, placeholder):
        adj = placeholder.E.cpu().numpy()
        if placeholder.node_mask is not None:
            n = torch.sum(placeholder.node_mask)
            adj = adj[:n, :n]
        nx_graph = nx.from_numpy_array(adj)
        # Delete weight on edges
        for _, _, data in nx_graph.edges(data=True):
            del data["weight"]
        node_features = placeholder.X.cpu().numpy()
        for node_idx in nx_graph.nodes():
            encoded_phenotype = node_features[node_idx]
            phenotype = PHENOTYPE_DECODER[encoded_phenotype]
            nx_graph.nodes[node_idx]["phenotype"] = phenotype

        cell_graph = cls(nx_graph)
        return cell_graph

    def to_placeholder(self, dataset_infos):
        tg_graph = self.to_torch_geometric()
        placeholder = to_dense(tg_graph, dataset_infos)
        placeholder = placeholder.collapse(collapse_charges=None)

        return placeholder

    def is_isomorphic(self, other_cg):
        return nx.is_isomorphic(
            self,
            other_cg,
            node_match=lambda x, y: x["phenotype"] == y["phenotype"],
        )

    def get_phenotypes_list(self):
        return [self.nodes[node]["phenotype"] for node in self.nodes()]

    def get_cell_phenotype_from_idx(self, idx):
        return self.nodes[idx]["phenotype"]

    @property
    def map_phenotype_to_color(self):
        return {
            "B": "c",
            "T": "b",
            "Epithelial": "k",
            "Fibroblast": "#C4A484",  # light brown
            "Myofibroblast": "g",
            "CD38+ Lymphocyte": "#FEE12B",  # yellow (towards gold)
            "Macrophages/Granulocytes": "C3",  # "r",
            "Marker": "0.9",  # grey
            "Endothelial": "0.75",  # grey
        }
        # map_phenotype_to_color = {
        #     "B": "#204474",  # dark blue
        #     "T": "#88B3CF",  # light blue
        #     "Epithelial": "#9E302B",  # red
        #     "Fibroblast": "#B696B5",  # violet
        #     "Myofibroblast": "#4EA48E",  # emerald
        #     "CD38+ Lymphocyte": "#C1D661",  # green automn
        #     "Macrophages/Granulocytes": "#A2CDB9",  # jade
        #     "Marker": "#F8E173",  # yellow
        #     "Endothelial": "#F4E0CA",  # beige
        # }
        # # "#E8935D", orange

    def set_pos(self, pos=None):
        if self.get_pos():
            raise ValueError("Positions already set")
        elif pos is None:
            positions = nx.spring_layout(self)
        else:
            positions = pos
        nx.set_node_attributes(self, positions, "pos")

    def get_pos(self):
        return nx.get_node_attributes(self, "pos")

    def plot_graph(
        self,
        has_legend=True,
        node_size=50,
        save_path=None,
        black_border=True,
        no_edges=False,
        fontsize=12,
        verbose=True,
    ):
        node_colors = [
            self.map_phenotype_to_color[phenotype]
            for phenotype in nx.get_node_attributes(self, "phenotype").values()
        ]

        # Positons
        if not self.get_pos():
            self.set_pos()
        positions = self.get_pos()

        # Plot graph
        plt.figure()
        nx.draw_networkx_nodes(
            self,
            # with_labels=False,
            pos=positions,
            node_size=node_size,
            node_color=node_colors,
            # font_weight=font_weight,
            edgecolors="k" if black_border else None,
        )
        if not no_edges:
            "here"
            nx.draw_networkx_edges(
                self,
                pos=positions,
                edge_color="k",
                width=1,
            )

        # Create a legend
        if has_legend:
            # Set LaTeX as the text renderer
            rc("text", usetex=True)
            rc("font", size=fontsize)
            legend_labels = list(self.map_phenotype_to_color.keys())
            legend_handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=10,
                    label=label,
                )
                for label, color in self.map_phenotype_to_color.items()
            ]
            plt.legend(
                handles=legend_handles,
                labels=legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.20),
                ncol=len(legend_labels) / 3,
            )

        plt.axis("off")  # delete black square around

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")
            if verbose:
                print(f"Saved graph plot to {save_path}")
            plt.close()
        else:
            plt.show()

    def plot_graph_with_tls_edges(
        self,
        save_path=None,
        has_legend=True,
        fontsize=12,
    ):
        blur_color = "0.9"
        node_colors = {
            node_idx: (
                self.map_phenotype_to_color[phenotype]
                if phenotype in ["B", "T"]
                else "w"
            )
            for node_idx, phenotype in nx.get_node_attributes(self, "phenotype").items()
        }

        nodes_positions = nx.get_node_attributes(self, "pos")
        if not nodes_positions:
            nodes_positions = nx.spring_layout(
                self, seed=0
            )  # needed for reproducibility
        for color in set(node_colors.values()):
            nodes_with_color = [
                node_idx
                for node_idx, node_color in node_colors.items()
                if node_color == color
            ]
            nx.draw_networkx_nodes(
                self,
                pos=nodes_positions,
                nodelist=nodes_with_color,
                node_color=color,
                edgecolors=blur_color if color == "w" else "k",
                node_size=100,
            )

        edge_types = []
        map_edge_type_to_color = {
            "ignore": blur_color,
            "alpha": "k",
            "gamma_0": "#1f77b4",
            "gamma_1": "#ff7f0e",
            "gamma_2": "#2ca02c",
            "gamma_3": "#d62728",
            "gamma_4": "#9467bd",
            "gamma_5": "#8c564b",
        }

        for edge in self.edges:
            edge_types.append(self.classify_TLS_edge(edge))
        edge_colors = [map_edge_type_to_color[edge_type] for edge_type in edge_types]
        edge_widths = [2 if edge_type != "ignore" else 0.5 for edge_type in edge_types]
        h_edges = nx.draw_networkx_edges(
            self,
            pos=nodes_positions,
            edge_color=edge_colors,
            width=edge_widths,
        )

        # # Legend
        from matplotlib.lines import Line2D

        def make_line(clr, **kwargs):
            return Line2D([], [], color=clr, linewidth=2, **kwargs)

        labels = []
        proxies = []
        for edge_type in [
            "alpha",
            "gamma_0",
            "gamma_1",
            "gamma_2",
            "gamma_3",
            "gamma_4",
            "gamma_5",
        ]:
            # get edge with that edge type
            try:
                edge_idx = edge_types.index(edge_type)
                edge_color = edge_colors[edge_idx]
                labels.append("$\\" + edge_type + "$")
                proxies.append(make_line(edge_color))
            except:
                continue  # that type does not exists in this graph

        plt.rcParams.update(
            {
                "text.latex.preamble": r"\renewcommand{\seriesdefault}{b}\boldmath",  # bold text and math in latex
                "text.usetex": True,
                "font.family": "Computer Modern Roman",
                "font.size": fontsize,
            }
        )

        plt.legend(
            handles=proxies,
            labels=labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.20),
            ncol=4,
        )

        plt.axis("off")  # delete square around

        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved graph plot to {save_path}")
        plt.close()

    def classify_TLS_edge(self, edge):
        allowed_cell_types = ["B", "T"]

        start_node, end_node = edge
        start_phenotype = self.nodes[start_node]["phenotype"]
        end_phenotype = self.nodes[end_node]["phenotype"]

        if (
            start_phenotype not in allowed_cell_types
            or end_phenotype not in allowed_cell_types
        ):
            edge_type = "ignore"

        elif start_phenotype == end_phenotype:
            edge_type = "alpha"
        else:
            b_cell = start_node if start_phenotype == "B" else end_node
            num_of_b_neighbors = len(
                [
                    node
                    for node in self.neighbors(b_cell)
                    if self.nodes[node]["phenotype"] == "B"
                ]
            )
            edge_type = f"gamma_{num_of_b_neighbors}"

        return edge_type

    def compute_tls_features(
        self,
        a_max: int = 5,
        min_num_gamma_edges: int = 0,
        verbose: bool = True,
    ):
        """Compute TLS feature metric from https://arxiv.org/pdf/2310.06661.pdf.

        Args:
            graph (nx.Graph): Graph to compute the TLS features. Should have a "phenotype" attribute with some nodes labeled as "B" and "T".
            a_max (int, optional): Maximum a of k(a) to consider. Defaults to 5.
            min_num_gamma_edges (int, optional): Minimum number of gamma edges to consider the

        Returns:
            dict: Dictionary with the TLO features.
        """

        if self.is_directed():
            raise ValueError("Graph should be undirected.")

        graph_phenotypes = nx.get_node_attributes(self, "phenotype")
        nodes_to_remove = [
            node
            for node, phenotype in graph_phenotypes.items()
            if phenotype != "B" and phenotype != "T"
        ]
        bt_subgraph = copy.deepcopy(self)
        bt_subgraph.remove_nodes_from(nodes_to_remove)
        total_num_edges = bt_subgraph.number_of_edges()

        # Get alpha and gamma edges count
        num_edge_types_idxs = self.get_edges_idxs_by_tlo_type(a_max)

        # Compute TLO features
        denominator = total_num_edges - num_edge_types_idxs["alpha"]
        # If the number of gamma edges are too few, then the feature estimation
        # is unreliable.
        if denominator < min_num_gamma_edges:
            tlo_dict = {f"k_{a}": None for a in range(a_max + 1)}
            if verbose:
                print("WARNING: too few gamma edges. TLO feature set to -1.")
        else:
            tlo_dict = {}
            if denominator == 0:
                tlo_dict.update({f"k_{a}": 0.0 for a in range(a_max + 1)})
            else:
                k = 1.0
                for a in range(a_max + 1):
                    k -= num_edge_types_idxs[f"gamma_{a}"] / denominator
                    # Due to precision errors, sometimes the k computed is negative. To avoid
                    # this, we clip it to 0 in those cases. The rounding is performed to 4
                    # decimal cases, as there is no meaning on the remaining decimal cases.
                    tlo_dict.update({f"k_{a}": max(0.0, round(k, 4))})

        return tlo_dict

    def get_edges_idxs_by_tlo_type(self, a_max: int):
        """Provided a graph, count

        Args:
            graph (nx.Graph): _description_
            a_max (int): _description_

        Returns:
            _type_: _description_
        """

        num_edges_by_type = {f"gamma_{a}": 0 for a in range(a_max + 1)}
        num_edges_by_type["alpha"] = 0

        for edge in self.edges:
            edge_type = self.classify_TLS_edge(edge)
            if edge_type in num_edges_by_type:
                num_edges_by_type[edge_type] += 1

        return num_edges_by_type
