import copy
from collections import Counter
from typing import List

import torch.nn as nn
import networkx as nx
import numpy as np
import scipy.sparse as sp
import wandb
from torchmetrics import MeanMetric, MaxMetric, Metric, MeanAbsoluteError
import torch
from torch import Tensor

from ConStruct.utils import PlaceHolder
from ConStruct.metrics.metrics_utils import (
    counter_to_tensor,
    wasserstein1d,
    total_variation1d,
)
from ConStruct.metrics.spectre_utils import SpectreSamplingMetrics, is_planar_graph
from ConStruct.datasets.tls_dataset import CellGraph
from ConStruct.analysis.dist_helper import compute_mmd, emd, gaussian_tv


class TLSSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, train_dataloader, val_dataloader, tls_type):
        super().__init__(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            compute_emd=False,
            metrics_list=[
                "degree",
                "clustering",
                "orbit",
                "spectre",
                "wavelet",
                "planar",
            ],
        )
        self.train_cell_graphs = self.loader_to_cell_graphs(train_dataloader)
        self.val_cell_graphs = self.loader_to_cell_graphs(val_dataloader)

        self.cell_graph_valid_fn = self.is_cell_graph_valid(tls_type=tls_type)

        self.mean_tls_validity = MeanMetric()

    def is_cell_graph_valid(self, tls_type):
        if tls_type == "high_tls":
            return lambda cg: cg.has_high_TLS() and is_planar_graph(cg)
        elif tls_type == "low_tls":
            return lambda cg: cg.has_low_TLS() and is_planar_graph(cg)
        else:
            raise ValueError(f"Invalid tls_type : {tls_type} (TLS Sampling Metrics)")

    def loader_to_cell_graphs(self, loader):
        cell_graphs = []
        for batch in loader:
            for tg_graph in batch.to_data_list():
                cell_graph = CellGraph.from_torch_geometric(tg_graph)
                cell_graphs.append(cell_graph)

        return cell_graphs

    def forward(self, generated_graphs: list, current_epoch, local_rank):
        to_log = super().forward(generated_graphs, current_epoch, local_rank)
        if local_rank == 0:
            print(
                f"Computing TLS sampling metrics between {sum([placeholder.X.shape[0] for placeholder in generated_graphs])} generated graphs and {len(self.val_graphs)}"
            )

        generated_cell_graphs = []
        for batch in generated_graphs:
            graph_placeholders = batch.split()
            for placeholder in graph_placeholders:
                cell_graph = CellGraph.from_placeholder(placeholder)
                generated_cell_graphs.append(cell_graph)

        # TLS features
        if local_rank == 0:
            print("Computing TLS features stats...")

        device = self.mean_tls_validity.device
        self.mean_tls_validity(
            tls_validity_ratio(generated_cell_graphs, self.cell_graph_valid_fn).to(
                device
            )
        )
        to_log["tls_metrics/mean_tls_validity"] = (
            self.mean_tls_validity.compute().item()
        )
        tls_stats = compute_tls_stats(
            generated_cell_graphs,
            self.val_cell_graphs,
            bins=100,
            compute_emd=self.compute_emd,
        )
        for key, value in tls_stats.items():
            to_log[f"tls_metrics/{key}"] = value
            if wandb.run:
                wandb.run.summary[f"tls_metrics/{key}"] = value

        # Isomorphic vs unique?
        if local_rank == 0:
            print("Computing uniqueness and isomorphic for cell graphs...")
            frac_novel = eval_fraction_novel_cell_graphs(
                generated_cell_graphs=generated_cell_graphs,
                train_cell_graphs=self.train_cell_graphs,
            )
            (
                frac_unique,
                frac_unique_and_novel,
                frac_unique_and_novel_valid,
            ) = eval_fraction_unique_novel_valid_cell_graphs(
                generated_cell_graphs=generated_cell_graphs,
                train_cell_graphs=self.train_cell_graphs,
                valid_cg_fn=self.cell_graph_valid_fn,
            )
        to_log.update(
            {
                "tls_metrics/frac_novel": frac_novel,
                "tls_metrics/frac_unique": frac_unique,
                "tls_metrics/frac_unique_and_novel": frac_unique_and_novel,
                "tls_metrics/frac_unique_and_novel_valid": frac_unique_and_novel_valid,
            }
        )

        if local_rank == 0:
            tls_sampling_metrics_log = {
                metric: value
                for metric, value in to_log.items()
                if "tls_metrics" in metric
            }
            print(f"TLS sampling statistics: {tls_sampling_metrics_log}")

        return to_log

    def reset(self):
        self.mean_tls_validity.reset()
        super().reset()


def tls_validity_ratio(generated_graphs: List[PlaceHolder], valid_cg_fn):
    cg_validities = [int(valid_cg_fn(cg)) for cg in generated_graphs]
    return torch.tensor(cg_validities)


# specific for cell graphs (isomorphism function is of cell graphs)
def eval_fraction_novel_cell_graphs(generated_cell_graphs, train_cell_graphs):
    count_non_novel = 0
    for gen_cg in generated_cell_graphs:
        for train_cg in train_cell_graphs:
            if nx.faster_could_be_isomorphic(train_cg, gen_cg):
                if gen_cg.is_isomorphic(train_cg):
                    count_non_novel += 1
                    break
    return 1 - count_non_novel / len(generated_cell_graphs)


# specific for cell graphs (isomorphism function is of cell graphs)
def eval_fraction_unique_novel_valid_cell_graphs(
    generated_cell_graphs,
    train_cell_graphs,
    valid_cg_fn,
):
    count_non_unique = 0
    count_not_novel = 0
    count_not_valid = 0
    for cg_idx, gen_cg in enumerate(generated_cell_graphs):
        is_unique = True
        for gen_cg_seen in generated_cell_graphs[:cg_idx]:
            if nx.faster_could_be_isomorphic(gen_cg_seen, gen_cg):
                # we also need to consider phenotypes of nodes
                if gen_cg.is_isomorphic(gen_cg_seen):
                    count_non_unique += 1
                    is_unique = False
                    break
        if is_unique:
            is_novel = True
            for train_cg in train_cell_graphs:
                if nx.faster_could_be_isomorphic(train_cg, gen_cg):
                    if gen_cg.is_isomorphic(train_cg):
                        count_not_novel += 1
                        is_novel = False
                        break
            if is_novel:
                if not valid_cg_fn(gen_cg):
                    count_not_valid += 1

    frac_unique = 1 - count_non_unique / len(generated_cell_graphs)
    frac_unique_non_isomorphic = frac_unique - count_not_novel / len(
        generated_cell_graphs
    )
    frac_unique_non_isomorphic_valid = (
        frac_unique_non_isomorphic - count_not_valid / len(generated_cell_graphs)
    )

    return (
        frac_unique,
        frac_unique_non_isomorphic,
        frac_unique_non_isomorphic_valid,
    )


def compute_tls_stats(generated_cell_graphs, val_cell_graphs, bins, compute_emd):
    """Compute TLS features for a set of graphs.

    Args:
        generated_cell_graphs (list): List of CellGraphs to compute the TLS features.
        val_cell_graphs (list): List of CellGraphs to compute the TLS features.

    Returns:

    """

    # Extract TLS features
    generated_tls_hists = cell_graphs_to_TLS_features_hists(generated_cell_graphs, bins)
    val_tls_features_hists = cell_graphs_to_TLS_features_hists(val_cell_graphs, bins)

    # Compute TLS features stats
    tls_stats = {}
    for key in generated_tls_hists.keys():
        generated_sample = [generated_tls_hists[key]]
        val_sample = [val_tls_features_hists[key]]
        if compute_emd:
            mmd_dist = compute_mmd(
                val_sample,
                generated_sample,
                kernel=emd,
            )
        else:
            mmd_dist = compute_mmd(
                val_sample,
                generated_sample,
                kernel=gaussian_tv,
            )
        tls_stats[key] = mmd_dist

    return tls_stats


def cell_graphs_to_TLS_features_hists(cell_graphs: List[CellGraph], bins):
    # Compute TLS features
    tls_features_list = []
    for cell_graph in cell_graphs:
        tls_features = cell_graph.compute_tls_features()
        tls_features_list.append(tls_features)

    # Group TLS features across k
    tls_features_grouped = {}
    for key in tls_features_list[0].keys():
        tls_features_grouped[key] = [
            tls_features[key] for tls_features in tls_features_list
        ]

    # Generate histograms
    tls_hists = {}
    for key in tls_features_grouped.keys():
        values_list = tls_features_grouped[key]
        tls_hists[key], _ = np.histogram(
            values_list, bins=bins, range=(0, 1), density=False
        )

    return tls_features_grouped
