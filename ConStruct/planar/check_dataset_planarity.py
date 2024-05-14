try:
    import graph_tool as gt
except ModuleNotFoundError:
    print("Graph tool not found, non molecular datasets cannot be used")
import os
import pathlib


import hydra
import json
import networkx as nx
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
import torch_geometric as tg
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm


from ConStruct.analysis.visualization import Visualizer
from ConStruct.utils import PlaceHolder
from ConStruct import utils


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.train.seed)
    dataset_config = cfg.dataset

    if dataset_config.name == "planar":
        from ConStruct.datasets.spectre_dataset import (
            PlanarDataModule,
            SpectreDatasetInfos,
        )

        datamodule = PlanarDataModule(cfg)
        dataset_infos = SpectreDatasetInfos(datamodule)

    elif dataset_config.name == "qm9":
        from ConStruct.datasets.qm9_dataset import QM9DataModule, QM9Infos

        datamodule = QM9DataModule(cfg)
        dataset_infos = QM9Infos(datamodule=datamodule, cfg=cfg)
    elif dataset_config.name == "guacamol":
        from ConStruct.datasets.guacamol_dataset import (
            GuacamolDataModule,
            GuacamolInfos,
        )

        datamodule = GuacamolDataModule(cfg)
        dataset_infos = GuacamolInfos(datamodule=datamodule, cfg=cfg)
    elif dataset_config.name == "moses":
        from ConStruct.datasets.moses_dataset import MosesDataModule, MosesInfos

        datamodule = MosesDataModule(cfg)
        dataset_infos = MosesInfos(datamodule=datamodule, cfg=cfg)
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg.dataset))

    datamodules = {
        "train": datamodule.train_dataloader(),
        "val": datamodule.val_dataloader(),
        "test": datamodule.test_dataloader(),
    }

    visualizer = Visualizer(dataset_infos=dataset_infos)

    non_planar_smiles = {}
    print(f"Dataset loaded: {cfg.dataset.name}")
    for datamodule_str, datamodule in datamodules.items():
        print(f"Starting analysing {datamodule_str} datamodule.")
        dataset_idx = 0
        for batch_idx, batch in enumerate(tqdm(datamodule)):
            for graph in batch.to_data_list():
                nx_graph = to_networkx(graph, to_undirected=True)
                assert (
                    to_dense_adj(graph.edge_index).numpy()
                    == nx.to_numpy_array(nx_graph)
                ).all()
                if not nx.is_planar(nx_graph):
                    print(
                        f"Found non-planar graph. Edge index: {graph.edge_index}. Num nodes: {graph.num_nodes}"
                    )
                    saving_path = os.path.join(
                        os.getcwd(),
                        f"{dataset_config.name}_non_planar_graphs_{datamodule_str}_{dataset_idx}",
                    )

                    # put graph in required format for visualization
                    dense_graph = utils.to_dense(graph, dataset_infos)
                    graph_to_plot = dense_graph.collapse(dataset_infos.collapse_charges)
                    visualizer.visualize(
                        path=saving_path,
                        graphs=graph_to_plot,
                        atom_decoder=(
                            dataset_infos.atom_decoder
                            if hasattr(dataset_infos, "atom_decoder")
                            else None
                        ),
                        num_graphs_to_visualize=1,
                    )
                    # Storing smiles as well
                    if dataset_infos.is_molecular:
                        non_planar_smiles[f"{datamodule_str}_{dataset_idx}"] = (
                            graph.smiles
                        )

                dataset_idx += 1

    # Write non planar smiles to file
    if dataset_infos.is_molecular:
        with open(f"non_planar_smiles_{dataset_config.name}.json", "w") as outfile:
            json.dump(non_planar_smiles, outfile)

    print(f"Found {len(non_planar_smiles)} non planar graphs")


if __name__ == "__main__":
    main()
